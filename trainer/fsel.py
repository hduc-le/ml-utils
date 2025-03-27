import logging
import os
import pickle
import uuid
from typing import Any, Dict, List, Tuple, Union

import IPython
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _get_shap_summary_figure(
    shap_values: np.array, data: np.array, feature_names: List[str]
):
    """
    Generate SHAP summary plot.

    Args:
        shap_values (np.array): SHAP values.
        data (np.array): data features.
        feature_names (List[str]): Feature names.

    """
    shap.summary_plot(
        shap_values,
        data,
        show=False,
        feature_names=feature_names,
        max_display=len(feature_names),
    )
    fig_path = f"lgbm_shap_summary_plot_{str(uuid.uuid4())[:8]}.png"
    plt.savefig(fig_path)
    try:
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)
    except Exception as e:
        logger.error(f"Fail to log shap summary plot to mlflow: {e}")
        os.remove(fig_path)


def _get_shap_importance(
    model: lgb.Booster, X_valid: np.array, show_summary_plot: bool = False
) -> Tuple[pd.DataFrame, np.array]:
    """
    Calculate SHAP feature importance for a given validation dataset and model.

    Args:
        X_valid (numpy.array): Validation dataset.
        model (lgb.Booster): Trained machine learning model compatible with SHAP.
        show_summary_plot (bool): If True, show SHAP summary plot.
    Returns:
        DataFrame: DataFrame containing feature names and corresponding SHAP scores.
    """
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_valid)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Take mean absolute SHAP values across samples
    avg_shap_values = np.abs(shap_values).mean(axis=0)

    # Create DataFrame to store feature importance
    feature_impr = pd.DataFrame(
        {"feature": model.feature_name(), "score": avg_shap_values}
    )

    # Sort features by their SHAP scores
    feature_impr = feature_impr.sort_values(by="score", ascending=False)

    if show_summary_plot:
        _get_shap_summary_figure(
            shap_values=shap_values,
            data=X_valid,
            feature_names=model.feature_name(),
        )

    return feature_impr, shap_values


def _get_feature_importance(
    model: lgb.Booster, importance_type: str
) -> pd.DataFrame:
    """
    Get feature importance.

    Args:
        model (lgb.Booster): The trained LightGBM model.
        importance_type (str): Type of feature importance.

    Returns:
        pd.DataFrame: DataFrame containing feature importance.

    """
    return pd.DataFrame(
        {
            "feature": model.feature_name(),
            "score": model.feature_importance(importance_type=importance_type),
        }
    ).sort_values(by="score", ascending=False)


class LGBMFeatureSelector:
    """
    A class to perform iterative feature selection for a LightGBM model on a binary classification task.
    The selection is based on reducing a percentage of features after each training turn and tracking the model's
    performance using either user-defined or default metrics.

    Parameters:
    ----------
    lgbm_params : Dict
        The LightGBM model parameters.
    train_set : Tuple[np.ndarray, np.ndarray] or Tuple[pd.DataFrame, pd.Series]
        A tuple of training features and labels.
    eval_set : Tuple[np.ndarray, np.ndarray] or Tuple[pd.DataFrame, pd.Series]
        A tuple of validation features and labels.
    test_set : Tuple[np.ndarray, np.ndarray] or Tuple[pd.DataFrame, pd.Series]
        A tuple of test features and labels.
    feature_columns : List[str]
        A list of feature column names used for training.
    categorical_columns : List[str]
        A list of categorical feature column names.
    target_column : str
        The name of the target column.
    feature_importance_type: str
        The type of feature importance to use for selection (e.g., "shap", "lgbm").
    drop_perc : float
        The feature decreasing rate in each iteration (percentage of features to remove in each turn).
    min_features : int
        The minimum number of features to remain in the model. The iteration stops when the number of features
        falls below this threshold.
    compute_metrics : Callable, optional
        A user-defined function that computes and returns a dictionary of metrics given predictions and ground truth.
        If not provided, a default function will be used that computes AUC (ROC), AUC PR, and log loss.
    primary_metric : str
        The primary metric to track for performance drops (e.g., "AUC").

    Methods:
    -------
    train_model(features):
        Trains a LightGBM model using the provided features and returns the model, best iteration,
        and predictions on train, validation, and test sets.

    run(output_dir):
        Performs the iterative feature selection process, reducing the number of features by drop_perc% after each turn.
        At each turn, the model is trained, metrics are recorded, and the primary metric change is tracked.
        The process continues until the number of features is below `min_features`. The function returns a summary
        report as a pandas DataFrame.

    save_metadata():
        Saves the metadata (i.e., training iterations, feature sets, and metrics) to a pickle file.

    load_metadata(path):
        Loads the metadata from a previously saved pickle file.
    """

    def __init__(
        self,
        lgbm_params: Dict = None,
        train_set: Union[
            Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]
        ] = None,
        eval_set: Union[
            Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]
        ] = None,
        test_set: Union[
            Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]
        ] = None,
        feature_columns: List[str] = None,
        categorical_columns: List[str] = None,
        target_column: str = "label",
        feature_importance_type: str = "shap",
        drop_perc: float = 0.05,
        min_features: int = 30,
        primary_metric: str = "AUC",
        compute_metrics: Any = None,
        run_name: str = None,
        experiment_name: str = None,
        tracking_uri: str = None,
        disable_display: bool = False,
    ):
        # Model params
        self.lgbm_params = lgbm_params
        self.train_data, self.train_label = train_set
        self.valid_data, self.valid_label = eval_set
        self.test_data, self.test_label = test_set
        self._handle_numpy_inputs()

        # Feature params
        self.feature_columns = feature_columns
        if self.feature_columns is None:
            self.feature_columns = self.train_data.columns.tolist()
        self.categorical_columns = categorical_columns
        self._handle_categorical_columns()

        self.target_column = target_column
        self.feature_importance_type = feature_importance_type

        # Experiment params
        self.drop_perc = drop_perc
        self.min_features = min_features

        # MLflow tracking params
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._init_mlflow()

        # Metrics and logging params
        self.primary_metric = primary_metric
        self.compute_metrics = (
            compute_metrics
            if compute_metrics is not None
            else self.default_compute_metrics
        )
        self.metadata = {}
        self.summary = pd.DataFrame(
            columns=[
                "Turn",
                "Start Num Features",
                "End Num Features",
                "Start Num Cat Features",
                "End Num Cat Features",
                "Best Iter",
            ]
        )
        self._disable_display = disable_display

    def _init_mlflow(self):
        """Initialize MLflow tracking URI and experiment."""
        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        logger.info(f"MLflow experiment name: {self.experiment_name}")
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create experiment if it does not exist
        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

    def _handle_numpy_inputs(self):
        """
        Check if the inputs are numpy arrays and convert them to pandas DataFrames.
        """
        if isinstance(self.train_data, np.ndarray):
            self.train_data = pd.DataFrame(
                self.train_data, columns=self.feature_columns
            )
        if isinstance(self.valid_data, np.ndarray):
            self.valid_data = pd.DataFrame(
                self.valid_data, columns=self.feature_columns
            )
        if isinstance(self.test_data, np.ndarray):
            self.test_data = pd.DataFrame(
                self.test_data, columns=self.feature_columns
            )

        if isinstance(self.train_label, np.ndarray):
            self.train_label = pd.Series(
                self.train_label, name=self.target_column
            )
        if isinstance(self.valid_label, np.ndarray):
            self.valid_label = pd.Series(
                self.valid_label, name=self.target_column
            )
        if isinstance(self.test_label, np.ndarray):
            self.test_label = pd.Series(
                self.test_label, name=self.target_column
            )

    def _handle_categorical_columns(self):
        if self.categorical_columns is None:
            self.categorical_columns = self.train_data.select_dtypes(
                "object"
            ).columns.tolist()
        if self.categorical_columns:
            for col in self.categorical_columns:
                self.train_data[col] = self.train_data[col].astype("category")
                self.valid_data[col] = self.valid_data[col].astype("category")
                self.test_data[col] = self.test_data[col].astype("category")

    @staticmethod
    def default_compute_metrics(
        true: np.array, preds: np.array, prefix: str = ""
    ) -> Dict:
        """
        Default function to compute AUC (ROC), AUC PR, and log loss.

        Parameters:
        ----------
        true : np.array
            Ground truth (true labels).
        preds : np.array
            Predicted probabilities from the model.

        Returns:
        -------
        metrics : dict
            A dictionary containing AUC (ROC), AUC PR, and log loss.
        """
        return {
            f"{prefix}_AUC": roc_auc_score(true, preds),
            f"{prefix}_AUC_PR": average_precision_score(true, preds),
            f"{prefix}_Log_Loss": log_loss(true, preds),
        }

    @staticmethod
    def _log_df_to_mlflow(df: pd.DataFrame, name: str):
        """Log a DataFrame as a CSV file artifact to MLflow."""
        file_name = f"{name}.csv"
        df.to_csv(file_name, index=False)
        mlflow.log_artifact(file_name)
        os.remove(file_name)
        logger.info(f"Logged {file_name} to MLflow")

    def train_model(self, features: List[str], verbose_eval: int = 50):
        """
        Trains a LightGBM model using the specified features and datasets.

        Parameters:
        ----------
        features : List[str]
            A list of feature column names to use for training.

        Returns:
        -------
        model : lgb.Booster
            The trained LightGBM model.
        best_iter : int
            The best iteration determined by early stopping.
        train_preds : np.array
            The model predictions on the training set.
        valid_preds : np.array
            The model predictions on the validation set.
        test_preds : np.array
            The model predictions on the test set.
        """
        if self.categorical_columns:
            cat_features = [
                f for f in features if f in self.categorical_columns
            ]
        else:
            cat_features = "auto"

        train_dataset = lgb.Dataset(
            self.train_data[features],
            label=self.train_label,
            categorical_feature=cat_features,
        )
        valid_dataset = lgb.Dataset(
            self.valid_data[features],
            label=self.valid_label,
            categorical_feature=cat_features,
            reference=train_dataset,
        )
        test_X = self.test_data[features]

        model = lgb.train(
            self.lgbm_params,
            train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            callbacks=[
                lgb.log_evaluation(verbose_eval, show_stdv=True),
            ],
        )

        best_iter = model.best_iteration

        train_preds = model.predict(
            self.train_data[features], num_iteration=best_iter
        )
        valid_preds = model.predict(
            self.valid_data[features], num_iteration=best_iter
        )
        test_preds = model.predict(test_X, num_iteration=best_iter)

        return model, best_iter, train_preds, valid_preds, test_preds

    def run(self, output_dir: str = "fsel_output", **kwargs):
        """
        Executes the iterative feature selection process. In each iteration, the model is trained,
        metrics are computed using the provided or default `compute_metrics` function, and the number of features
        is reduced by drop_perc%. The process stops when the number of features falls below `min_features`. A summary report is generated and
        returned along with flagged iterations.

        Args:

        Returns:
        -------
        summary : pd.DataFrame
            A DataFrame summarizing the results of each training turn, including feature counts and custom metrics.
        """
        current_features = self.feature_columns
        turn = 0
        turn_logs = []

        os.makedirs(output_dir, exist_ok=True)

        with mlflow.start_run(run_name=self.run_name):
            while len(current_features) >= self.min_features:
                start_features = current_features.copy()
                start_num_features = len(start_features)
                start_cat_features = [
                    f for f in start_features if f in self.categorical_columns
                ]
                start_num_cat_features = len(start_cat_features)

                # Train the model and get predictions
                (
                    model,
                    best_iter,
                    train_preds,
                    valid_preds,
                    test_preds,
                ) = self.train_model(start_features, **kwargs)
                mlflow.lightgbm.log_model(model, f"model_{turn}")

                # Compute metrics using the user-defined or default function
                train_metrics = self.compute_metrics(
                    self.train_label, train_preds, prefix="Train"
                )
                valid_metrics = self.compute_metrics(
                    self.valid_label, valid_preds, prefix="Valid"
                )
                test_metrics = self.compute_metrics(
                    self.test_label, test_preds, prefix="Test"
                )

                # Get feature importances and reduce the feature set
                if self.feature_importance_type == "shap":
                    feature_importance_df, _ = _get_shap_importance(
                        model,
                        self.valid_data[current_features].sample(10_000),
                        show_summary_plot=False,
                    )
                elif self.feature_importance_type == "lgbm":
                    feature_importance_df = _get_feature_importance(
                        model, importance_type="gain"
                    )
                else:
                    raise ValueError(
                        "Invalid feature importance type. Must be 'shap' or 'lgbm'."
                    )

                feature_importance_df = feature_importance_df.sort_values(
                    by="score", ascending=False
                )

                # Reduce features by drop_perc%
                num_to_drop = max(
                    int(len(current_features) * self.drop_perc), 1
                )
                current_features = (
                    feature_importance_df["feature"]
                    .iloc[:-num_to_drop]
                    .tolist()
                )

                end_features = current_features.copy()
                end_num_features = len(end_features)
                end_cat_features = [
                    f for f in end_features if f in self.categorical_columns
                ]
                end_num_cat_features = len(end_cat_features)

                # Log the results of this turn
                log_row = {
                    "Turn": turn,
                    "Start Num Features": start_num_features,
                    "End Num Features": end_num_features,
                    "Start Num Cat Features": start_num_cat_features,
                    "End Num Cat Features": end_num_cat_features,
                    "Best Iter": best_iter,
                }
                # Add custom metrics to the log row
                for metrics in [train_metrics, valid_metrics, test_metrics]:
                    log_row.update(metrics)

                # Append the log row to the summary dataframe
                self.summary = pd.concat(
                    [self.summary, pd.DataFrame([log_row])], ignore_index=True
                ).fillna(-1)

                #
                if not self._disable_display:
                    IPython.display.clear_output()
                    IPython.display.display(
                        self.summary.style.background_gradient(
                            axis=0, cmap="BuGn", subset=("Best Iter")
                        )
                        .background_gradient(
                            axis=0,
                            cmap="BuPu",
                            subset=(f"Train_{self.primary_metric}"),
                        )
                        .background_gradient(
                            axis=0,
                            cmap="GnBu",
                            subset=(f"Valid_{self.primary_metric}"),
                        )
                        .background_gradient(
                            axis=0,
                            cmap="PuBu",
                            subset=(f"Test_{self.primary_metric}"),
                        )
                    )

                # Save metadata for later retrieval
                metadata_entry = {
                    "turn": turn,
                    "features_start": start_features,
                    "features_end": end_features,
                    "num_features_start": start_num_features,
                    "num_features_end": end_num_features,
                    "cat_features_start": start_cat_features,
                    "cat_features_end": end_cat_features,
                    "num_cat_feature_start": start_num_cat_features,
                    "num_cat_feature_end": end_num_cat_features,
                    "train_metrics": train_metrics,
                    "valid_metrics": valid_metrics,
                    "test_metrics": test_metrics,
                    "best_iter": best_iter,
                }
                turn_logs.append(metadata_entry)

                self.metadata["turn_logs"] = turn_logs
                # self.metadata["summary"] = self.summary

                self.save_metadata(os.path.join(output_dir, "metadata.pkl"))
                mlflow.log_dict(metadata_entry, f"metadata_{turn}.json")
                turn += 1

            mlflow.log_dict(self.metadata, "metadata.json")
            self._log_df_to_mlflow(self.summary, "summary")

        return self.summary

    def save_metadata(self, metadata_path: str):
        """
        Saves the metadata (i.e., training iterations, feature sets, and metrics) to a pickle file.
        """
        with open(metadata_path, "wb") as file:
            pickle.dump(self.metadata, file)

    @classmethod
    def load_metadata(cls, path: str):
        """
        Loads the metadata from a previously saved pickle file.

        Parameters:
        ----------
        path : str
            Path to the pickle file containing saved metadata.

        Returns:
        -------
        metadata : list
            The loaded metadata.
        """
        with open(path, "rb") as file:
            metadata = pickle.load(file)
        return metadata
