import os
from typing import Any, Callable, Dict

import mlflow
import pandas as pd
import yaml


def fetch_runs_by_name(
    run_name, experiment_name: str, tracking_uri: str, max_results: int = 1
) -> pd.DataFrame:
    """
    Fetch runs from an MLflow experiment.

    Args:
        run_name (str): The name of the run.
        experiment_name (str): The name of the experiment.
        tracking_uri (str): The URL of the MLflow tracking server.
        max_results (int): The maximum number of results to return.

    Returns:
        pd.DataFrame: A DataFrame containing the runs.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.get_experiment_by_name(
        experiment_name
    ).experiment_id
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName LIKE '{run_name}' AND status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=max_results,
    )
    if runs.empty:
        raise ValueError(f"No finished runs found with run name: {run_name}")

    return runs


class MLflowHelper:
    def __init__(
        self,
        experiment_name: str,
        tracking_url: str,
        artifact_location: str = None,
    ):
        """
        Initialize MLflowHelper for tracking experiments.

        Args:
            experiment_name (str): The name of the experiment.
            tracking_url (str): The URL of the MLflow tracking server.
            artifact_location (str): The artifact location.
        """
        self.experiment_name = experiment_name
        self.tracking_url = tracking_url
        self.artifact_location = artifact_location
        self.setup_mlflow()

    def setup_mlflow(self):
        """
        Setup MLflow for tracking experiments.

        Returns:
            None
        """
        mlflow.set_tracking_uri(self.tracking_url)
        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(
                self.experiment_name, artifact_location=self.artifact_location
            )
        mlflow.set_experiment(self.experiment_name)
        print(f"MLflow experiment: {self.experiment_name}")

    def start_run(self, run_name: str = None):
        """
        Start an MLflow run.

        Args:
            run_name (str): The name of the run.

        Returns:
            None
        """
        return mlflow.start_run(run_name=run_name)

    def end_run(self):
        """
        End the current MLflow run.

        Returns:
            None
        """
        return mlflow.end_run()

    def log_yaml(self, yaml_dict: dict, name: str):
        """
        Log a YAML dictionary as an artifact in MLflow.

        Args:
            yaml_dict (dict): A dictionary containing YAML data.
            name (str): The name of the artifact.

        Returns:
            None
        """
        file_name = f"{name}.yaml"
        with open(file_name, "w") as f:
            yaml.dump(yaml_dict, f)
        mlflow.log_artifact(file_name)
        os.remove(file_name)
        print(f"Logged {file_name}")

    def log_dataframe(self, df: pd.DataFrame, name: str):
        """
        Log a pandas DataFrame as an artifact in MLflow.

        Args:
            df (pd.DataFrame): The DataFrame to be logged.
            name (str): The name of the artifact.

        Returns:
            None
        """
        file_name = f"{name}.csv"
        df.to_csv(file_name, index=False)
        mlflow.log_artifact(file_name)
        os.remove(file_name)
        print(f"Logged {file_name}")

    def log_plot(
        self, args: Any, plot_func: Callable[..., Any], fp: str
    ) -> None:
        """
        Logs a plot to MLflow.

        Args:
            args (Any): The arguments to pass to the plot function.
            plot_func (Callable[..., Any]): The function that generates the plot.
            fp (str): The file path to save the logged plot.

        Returns:
            None
        """
        if not isinstance(args, tuple):
            args = (args,)

        fig = plot_func(*args)
        mlflow.log_figure(fig, fp)
        print(f"Logged {fp}")

    def log_figure(self, fig, name):
        """
        Log a plotly figure as an artifact in MLflow.

        Args:
            fig: The plotly figure to log.
            name: The name of the artifact.

        Returns:
            None
        """
        mlflow.log_figure(fig, name)
        print(f"Logged figure {name}")

    def get_model_uri_from_latest_run(
        self, run_name: str, experiment_name: str, tracking_url: str = None
    ) -> str:
        """
        Get the model URI from the latest run of a given experiment.

        Args:
            run_name (str): The name of the run.
            experiment_name (str): The name of the experiment.
            tracking_url (str): The URL of the MLflow tracking server (optional).

        Returns:
            str: The URI of the model.
        """
        if tracking_url is None:
            tracking_url = self.tracking_url

        mlflow.set_tracking_uri(tracking_url)
        experiment_id = mlflow.get_experiment_by_name(
            experiment_name
        ).experiment_id
        runs = mlflow.search_runs(
            experiment_ids=experiment_id,
            filter_string=f"tags.mlflow.runName like '{run_name}' and status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        model_uri = os.path.join(runs.iloc[0]["artifact_uri"], "model")
        return model_uri

    def get_artifact_uri_from_latest_run(
        self, run_name: str = None, experiment_name: str = None
    ) -> str:
        """
        Get the model URI from the latest run of a given experiment.

        Args:
            run_name (str): The name of the run.
            experiment_name (str): The name of the experiment.

        Returns:
            str: The URI of the model.
        """
        experiment_id = mlflow.get_experiment_by_name(
            experiment_name if experiment_name else self.experiment_name
        ).experiment_id
        runs = mlflow.search_runs(
            experiment_ids=experiment_id,
            filter_string=f"tags.mlflow.runName like '{run_name}' and status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs.iloc[0]["artifact_uri"]

    def log_params(self, params: Dict[str, Any]):
        """
        Log a dictionary of parameters.

        Args:
            params (Dict[str, Any]): Dictionary of parameters to log.

        Returns:
            None
        """
        mlflow.log_params(params)
        print(f"Logged parameters: {params}")

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log a dictionary of metrics.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics to log.

        Returns:
            None
        """
        mlflow.log_metrics(metrics)
        print(f"Logged metrics: {metrics}")

    def log_text(self, text: str, name: str):
        """
        Log a text file.

        Args:
            text (str): The text content to log.
            name (str): The name of the artifact.

        Returns:
            None
        """
        file_name = f"{name}.txt"
        with open(file_name, "w") as f:
            f.write(text)
        mlflow.log_artifact(file_name)
        os.remove(file_name)
        print(f"Logged {file_name}")

    def log_image(self, image_path: str, artifact_path: str):
        """
        Log an image file as an artifact in MLflow.

        Args:
            image_path (str): The path to the image file.
            artifact_path (str): The artifact path in MLflow.

        Returns:
            None
        """
        mlflow.log_artifact(image_path, artifact_path)
        print(f"Logged image {image_path} to {artifact_path}")
