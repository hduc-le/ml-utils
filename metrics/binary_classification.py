from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


def compute_bin_calibration_error(d: pd.DataFrame):
    """
    Compute the calibration error for a given bin.

    The calibration error is the absolute difference between the mean
    predicted probability and the mean actual label, scaled by the number
    of samples in the bin.

    Args:
        d (pd.DataFrame): A DataFrame containing the actual labels and
                          predicted probabilities for a specific bin.
                          It must have columns 'label' and 'prob'.

    Returns:
        float: The calibration error for the bin.
    """
    acc = d["label"].mean()
    conf = d["prob"].mean()
    return np.abs(acc - conf) * len(d)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_edges: List[float] = [0, 0.2, 0.4, 0.6, 0.8, 1],
    bin_labels: List[int] = [1, 2, 3, 4, 5],
) -> float:
    """
    Compute the expected calibration error.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted probabilities.
        bin_edges (List[float], optional): The bin edges for grouping. Defaults to [0, 0.2, 0.4, 0.6, 0.8, 1].
        bin_labels (List[int], optional): The bin labels. Defaults to [1, 2, 3, 4, 5].

    Returns:
        float: The expected calibration error.
    """
    df = pd.DataFrame({"label": y_true, "prob": y_pred})
    df["bin"] = pd.cut(df["prob"], bins=bin_edges, labels=bin_labels)
    return (
        df.groupby("bin").apply(compute_bin_calibration_error) / len(df)
    ).sum()


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> Dict[str, float]:
    """
    Compute evaluation metrics for binary classification.

    Args:
        y_true (np.ndarray): List of true labels.
        y_pred (np.ndarray): List of predicted probabilities.
        prefix (str): Prefix to add to metric names (optional).

    Returns:
        Dict[str, float]: Dictionary containing the computed metrics.
    """
    # Compute metrics
    auc_roc = roc_auc_score(y_true, y_pred)
    log_loss_score = log_loss(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)

    # Compute expected calibration error
    ece = compute_expected_calibration_error(y_true, y_pred)

    # Create and return dictionary of metrics
    return {
        f"{prefix}_auc": auc_roc,
        f"{prefix}_log_loss": log_loss_score,
        f"{prefix}_auc_pr": auc_pr,
        f"{prefix}_ece": ece,
    }


def plot_roc_curve(
    fpr: List[float], tpr: List[float], auc_score: float
) -> go.Figure:
    """
    Plot the ROC curve.
    :param fpr: false positive rate
    :param tpr: true positive rate
    :param auc_score: auc score
    :return: plotly figure
    """
    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            line_color="darkorange",
            line_width=2,
            name="ROC Curve (AUC = {:.2f})".format(auc_score),
        )
    )

    # Add diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line_color="navy",
            line_width=2,
            line_dash="dash",
            name="Diagonal",
        )
    )

    # Customize chart appearance
    fig.update_layout(
        title="Receiver Operating Characteristic Example",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.05],
        legend_title="",
        legend_orientation="h",
        legend_x=0.5,
        legend_y=1.1,
        width=600,
        height=600,
    )

    return fig


def plot_calibration_curve_with_count(
    y_true: np.ndarray, y_pred: np.ndarray, n_bin: int
) -> go.Figure:
    """
    Plot the calibration curve with count for each bin.
    :param y_true: ground truth labels
    :param y_pred: predicted probabilities
    :param n_bin: number of bins
    :return: plotly figure
    """
    # Calculate the true positive rate and predicted positive rate
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bin)

    # Calculate the counts for each bin
    counts = (
        pd.Series(
            pd.cut(
                y_pred,
                bins=np.linspace(0, 1, n_bin),
                labels=range(n_bin - 1),
            )
        )
        .value_counts()
        .to_dict()
    )

    # Create a list of counts for each bin
    bin_counts = [counts.get(i, 0) for i in range(n_bin - 1)]

    # Create a new figure
    fig = go.Figure(
        data=[
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                marker_symbol="circle",
                name="Calibration Curve",
                yaxis="y",
                offsetgroup=1,
            ),
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                line_dash="dash",
                name="Diagonal",
                yaxis="y",
                offsetgroup=1,
            ),
            go.Bar(
                x=prob_pred,
                y=bin_counts,
                name="Bar Data",
                yaxis="y2",
                offsetgroup=2,
                opacity=0.5,
            ),
            go.Scatter(
                x=prob_pred,
                y=bin_counts,
                mode="markers",
                marker_symbol="circle",
                yaxis="y2",
                offsetgroup=2,
            ),
        ],
        layout={
            "yaxis": {"title": "True Probability"},
            "yaxis2": {
                "title": "Sample Count",
                "overlaying": "y",
                "side": "right",
            },
        },
    )

    # Update the layout of the figure
    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Predicted Probability",
        yaxis_title="True Probability in Each Bin",
        legend_x=0.5,
        legend_y=1.2,
        width=600,
        height=600,
    )

    return fig
