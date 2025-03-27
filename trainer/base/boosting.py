import abc
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass
class ProblemTypeKeys:
    BINARY: str = "binary"
    MULTICLASS: str = "multiclass"
    MULTILABEL: str = "multilabel"
    REGRESSION: str = "regression"


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        train_set: Union[
            Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]
        ],
        eval_sets: List[
            Union[
                Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]
            ]
        ],
        test_set: Optional[
            Union[
                Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]
            ]
        ] = None,
        feature_name: List[str] = None,
        categorical_name: List[str] = None,
        output_dir: str = "/tmp/model_runs",
        **kwargs,
    ):
        # Feature and categorical handling
        self.feature_name = self._get_feature_names(train_set[0], feature_name)
        self.categorical_name = categorical_name or []
        self._validate_categorical_features()

        # Data processing
        self.X_train, self.y_train = self._process_data(*train_set)
        self.eval_sets = [self._process_data(X, y) for X, y in eval_sets]
        self.X_test, self.y_test = (
            self._process_data(*test_set) if test_set else (None, None)
        )
        if self.X_test is not None:
            logger.info("Test set is provided")
        # Problem type detection
        self.problem_type = self._determine_problem_type()
        logger.info(f"Problem type: {self.problem_type}")
        self.kwargs = kwargs
        logger.info(f"kwargs: {self.kwargs}")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output dir: {self.output_dir}")
        self.model = None

    def _process_data(self, X, y):
        assert isinstance(
            X, (pd.DataFrame, np.ndarray)
        ), "X must be a pandas DataFrame or numpy array."
        assert isinstance(
            y, (pd.Series, np.ndarray, pd.DataFrame)
        ), "y must be a pandas Series, DataFrame or numpy array."
        self._is_pd_data = isinstance(X, pd.DataFrame)
        # Convert numpy arrays to pandas objects; if y is multidimensional, assume multi-label and convert to DataFrame.
        if not self._is_pd_data:
            X = pd.DataFrame(X, columns=self.feature_name)
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                y = pd.Series(y)
            else:
                # Assume multi-label and create a DataFrame with default column names
                y = pd.DataFrame(
                    y, columns=[f"label_{i}" for i in range(y.shape[1])]
                )
        return X, y

    @staticmethod
    def _get_feature_names(X, feature_name):
        if feature_name is not None:
            return feature_name
        return (
            X.columns.tolist()
            if isinstance(X, pd.DataFrame)
            else [f"f{i}" for i in range(X.shape[1])]
        )

    def _validate_categorical_features(self):
        for cat in self.categorical_name:
            if cat not in self.feature_name:
                raise ValueError(
                    f"Categorical feature '{cat}' not found in feature names."
                )

    def _determine_problem_type(self):
        if (
            isinstance(self.y_train, pd.DataFrame)
            and self.y_train.shape[1] > 1
        ):
            return ProblemTypeKeys.MULTILABEL
        else:
            unique = np.unique(self.y_train)
            if len(unique) == 1:
                raise ValueError("All samples have the same target value.")
            return (
                ProblemTypeKeys.REGRESSION
                if np.issubdtype(self.y_train.to_numpy().dtype, np.floating)
                else ProblemTypeKeys.BINARY
                if len(unique) == 2
                else ProblemTypeKeys.MULTICLASS
            )

    @staticmethod
    def _init_mlflow(experiment_name: str, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        pass

    @abc.abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]):
        pass

    def load_model(self, path: str):
        """Load the model(s) from a local file or a remote URI."""
        if path.startswith("gs://"):
            return self._load_model_from_gcs_uri(path)
        else:
            return self._load_model_from_local_file(path)

    def save_model(self, path):
        if path.startswith("gs://"):
            return self._save_model_to_gcs_uri(path)
        else:
            return self._save_model_to_local_file(path)

    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics based on the problem type.

        For classification problems we assume that y_pred contains probabilities.
        For binary classification, y_pred is a 1D array with the probability for the positive class.
        For multiclass classification, y_pred is a 2D array with probabilities for each class.
        For multilabel classification, y_pred is a 2D array (samples x labels) with probabilities for each label.
        """
        metrics = {}

        if self.problem_type == ProblemTypeKeys.REGRESSION:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics["mae"] = mean_absolute_error(y_true, y_pred)

        elif self.problem_type in [
            ProblemTypeKeys.BINARY,
            ProblemTypeKeys.MULTICLASS,
        ]:
            y_pred_labels = np.argmax(y_pred, axis=1)
            metrics["accuracy"] = accuracy_score(y_true, y_pred_labels)

            if self.problem_type == ProblemTypeKeys.BINARY:
                metrics["f1_score"] = f1_score(y_true, y_pred_labels)
                metrics["auc"] = roc_auc_score(y_true, y_pred[:, 1])
            else:
                metrics["f1_score"] = f1_score(
                    y_true, y_pred_labels, average="macro"
                )
                # Specify multi_class="ovr" (one-vs-rest) and choose an averaging method.
                metrics["auc"] = roc_auc_score(
                    y_true, y_pred, multi_class="ovr", average="macro"
                )

        elif self.problem_type == ProblemTypeKeys.MULTILABEL:
            # Ensure y_true and y_pred are numpy arrays.
            y_true_np = (
                y_true.values if isinstance(y_true, pd.DataFrame) else y_true
            )
            y_pred_prob = (
                y_pred.values if isinstance(y_pred, pd.DataFrame) else y_pred
            )
            # For multilabel, threshold the probabilities at 0.5 to get binary predictions.
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            metrics["accuracy"] = accuracy_score(y_true_np, y_pred_binary)
            metrics["f1_score"] = f1_score(
                y_true_np, y_pred_binary, average="macro"
            )
            metrics["auc"] = roc_auc_score(
                y_true_np, y_pred_prob, average="macro"
            )

        return metrics

    def _save_model_to_local_file(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to local file: {path}")

    def _save_model_to_gcs_uri(self, path: str):
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
        with open(temp_path, "wb") as f:
            pickle.dump(self.model, f)

        bucket_name, blob_name = self._parse_gcs_uri(path)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_path)
        logger.info(f"Model saved to GCS: {path}")
        os.remove(temp_path)

    def _load_model_from_gcs_uri(self, path: str):
        bucket_name, blob_name = self._parse_gcs_uri(path)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
        blob.download_to_filename(temp_path)
        model = self._load_model_from_local_file(temp_path)
        logger.info(f"Model loaded from GCS: {path}")
        os.remove(temp_path)
        return model

    @staticmethod
    def _load_model_from_local_file(path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from local file: {path}")
        return model

    @staticmethod
    def _parse_gcs_uri(path: str):
        uri = path[len("gs://") :]
        parts = uri.split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        return bucket_name, blob_name
