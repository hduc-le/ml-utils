import logging
import os
from typing import Dict, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import shap
from sklearn.multioutput import MultiOutputClassifier

from .base import BaseTrainer, ProblemTypeKeys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class LGBMTrainer(BaseTrainer):
    def _handle_cat_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert object-typed categorical columns to numeric codes."""
        X = X.copy()
        for col in self.categorical_name:
            if col in X.columns and X[col].dtype == "object":
                # Convert to pandas Categorical and then take the integer codes.
                X[col] = X[col].astype("category").cat.codes
        return X

    def _process_data(self, X, y) -> pd.DataFrame:
        X, y = super()._process_data(X, y)
        X = self._handle_cat_features(X)
        return X, y

    def train(
        self,
        mlflow_run_name: str = None,
        mlflow_tracking_uri: str = "http://localhost:5000",
        mlflow_experiment_name: str = "LGBM_Experiment",
    ):
        """Train the LightGBM model(s) using the sklearn API, log parameters, metrics, and artifacts with MLflow."""
        self._init_mlflow(mlflow_experiment_name, mlflow_tracking_uri)
        mlflow.sklearn.autolog()
        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_params(self.kwargs)
            mlflow.log_params(
                {
                    "train_shape": self.X_train.shape,
                    "train_pos_rate": np.mean(self.y_train),
                }
            )
            mlflow.log_params(
                {
                    f"valid_{idx}_shape": X_val.shape
                    for idx, (X_val, _) in enumerate(self.eval_sets)
                }
            )
            mlflow.log_params(
                {
                    f"valid_{idx}_pos_rate": np.mean(y_val)
                    for idx, (_, y_val) in enumerate(self.eval_sets)
                }
            )
            if self.X_test is not None:
                mlflow.log_params(
                    {
                        "test_shape": self.X_test.shape,
                        "test_pos_rate": np.mean(self.y_test),
                    }
                )

            if self.problem_type == ProblemTypeKeys.MULTILABEL:
                # Use MultiOutputClassifier with a base LGBMClassifier for each label.
                base_model = lgb.LGBMClassifier(**self.kwargs)
                model = MultiOutputClassifier(base_model)
                model.fit(self.X_train, self.y_train)
                self.model = model

                # Log feature importance and SHAP values for each label's estimator.
                for idx, estimator in enumerate(model.estimators_):
                    label = self.y_train.columns[idx]
                    self._log_feature_importance(estimator, label=label)
                    self._log_shap_values(estimator, self.X_train, label=label)

            elif self.problem_type == ProblemTypeKeys.REGRESSION:
                model = lgb.LGBMRegressor(**self.kwargs)
                # If evaluation sets are provided, pass them to fit (they must be a list of (X_val, y_val))
                eval_set = [(self.X_train, self.y_train)] + (
                    [(X_val, y_val) for X_val, y_val in self.eval_sets]
                    if self.eval_sets
                    else None
                )
                model.fit(self.X_train, self.y_train, eval_set=eval_set)
                self.model = model
                self._log_feature_importance(self.model)
                self._log_shap_values(self.model, self.X_train)

            elif self.problem_type in [
                ProblemTypeKeys.BINARY,
                ProblemTypeKeys.MULTICLASS,
            ]:
                model = lgb.LGBMClassifier(**self.kwargs)
                eval_set = [(self.X_train, self.y_train)] + (
                    [(X_val, y_val) for X_val, y_val in self.eval_sets]
                    if self.eval_sets
                    else None
                )
                model.fit(self.X_train, self.y_train, eval_set=eval_set)
                self.model = model
                self._log_feature_importance(self.model)
                self._log_shap_values(self.model, self.X_train)

            # Evaluate on evaluation sets and log metrics.
            self._evaluate()

            # Save and log the model artifact.
            model_path = os.path.join(self.output_dir, "model.pkl")
            self.save_model(model_path)
            mlflow.sklearn.log_model(self.model, "sklearn")
            mlflow.lightgbm.log_model(self.model.booster_, "lightgbm")

            # Test the model if a test set is provided.
            self._test()

    def predict(self, X, **kwargs):
        _X = self._handle_cat_features(X) if isinstance(X, pd.DataFrame) else X
        return self.model.predict(_X, **kwargs)

    def predict_proba(self, X, **kwargs):
        _X = self._handle_cat_features(X) if isinstance(X, pd.DataFrame) else X
        if self.problem_type == ProblemTypeKeys.MULTILABEL:
            return (
                np.array(self.model.predict_proba(_X, **kwargs)).max(axis=2).T
            )
        return self.model.predict_proba(_X, **kwargs)

    def _evaluate(self):
        """Evaluate the model on each evaluation set and log metrics via MLflow."""
        train_preds = self.predict_proba(self.X_train)
        train_metrics = self.compute_metrics(self.y_train, train_preds)
        self._log_metrics(train_metrics, "train")
        logger.info(f"Train metrics: {train_metrics}")

        for idx, (X_val, y_val) in enumerate(self.eval_sets):
            preds = self.predict_proba(X_val)
            metrics = self.compute_metrics(y_val, preds)
            self._log_metrics(metrics, f"eval_{idx}")
            logger.info(f"Evaluation metrics on eval set {idx}: {metrics}")

    def _test(self):
        """Evaluate the model on the test set (if provided) and log metrics via MLflow."""
        if self.X_test is not None and self.y_test is not None:
            preds = self.predict_proba(self.X_test)
            metrics = self.compute_metrics(self.y_test, preds)
            self._log_metrics(metrics, "test")
            logger.info(f"Test metrics: {metrics}")
            return metrics
        else:
            logger.info("No test set provided.")
            return None

    def _log_feature_importance(self, estimator, label: Optional[str] = None):
        """Log a bar-plot of feature importance using Plotly and MLflow."""
        try:
            importance = estimator.booster_.feature_importance(
                importance_type="gain"
            )
            features = estimator.booster_.feature_name()
        except Exception:
            importance = estimator.feature_importances_
            features = self.feature_name

        df_imp = pd.DataFrame({"feature": features, "importance": importance})
        df_imp.sort_values("importance", ascending=False, inplace=True)
        title = "Feature Importance" + (f" ({label})" if label else "")

        fig = px.bar(df_imp, x="feature", y="importance", title=title)
        fig.update_layout(xaxis_tickangle=-45)

        fi_path = os.path.join(
            self.output_dir,
            f"feature_importance_{label if label else 'model'}.html",
        )
        fig.write_html(fi_path)
        mlflow.log_artifact(fi_path, artifact_path="feature_importance")

    def _log_shap_values(
        self, estimator, X: pd.DataFrame, label: Optional[str] = None
    ):
        """Compute and log a SHAP beeswarm summary plot (using Plotly) as an artifact via MLflow."""
        X_sample = X.iloc[:1_000].copy()
        try:
            # Try using the underlying booster.
            explainer = shap.TreeExplainer(estimator.booster_)
        except Exception:
            explainer = shap.TreeExplainer(estimator)

        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            # For multiclass models, take the first class for visualization.
            shap_values = shap_values[0]

        shap.summary_plot(shap_values, X_sample, show=False)
        file_path = os.path.join(
            self.output_dir, f"shap_summary_{label if label else 'model'}.png"
        )
        plt.savefig(file_path)
        mlflow.log_artifact(file_path, "shap_plot")

    @staticmethod
    def _log_metrics(metrics: Dict[str, float], prefix: str = ""):
        for key, value in metrics.items():
            mlflow.log_metric(f"{prefix}_{key}", value)
