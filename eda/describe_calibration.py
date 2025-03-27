from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import pandas as pd 
import numpy as np

import numpy as np
import logging

import plotly.subplots as sp

import plotly.graph_objects as go
import ipywidgets as widgets

def plot_calibration_curve_with_count(y_true: np.ndarray, y_pred: np.ndarray, n_bin: int, title: str = "Calibration Curve", fig_size: tuple = (800, 800)) -> None:
    # Calculate the true positive rate and predicted positive rate
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bin)

    # Calculate the counts for each bin
    counts = pd.Series(pd.cut(y_pred, bins=np.linspace(0, 1, n_bin), labels=range(n_bin-1))).value_counts().to_dict()

    # Create a list of counts for each bin
    bin_counts = [counts.get(i, 0) for i in range(n_bin-1)]

    # Calculate the Expected Calibration Error (ECE)
    ece = expected_calibration_error(y_pred=y_pred, y_true=y_true, n_bins=n_bin)

    # Create the calibration curve figure
    fig = go.Figure(
        data=[
            go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', marker_symbol='circle', 
                       name='Calibration Curve', yaxis='y', offsetgroup=1),
            go.Scatter(x=[0, 1], y=[0, 1], line_dash='dash', name='Diagonal', yaxis='y', offsetgroup=1),
            go.Bar(x=prob_pred, y=bin_counts, name='Bar Data',yaxis='y2', offsetgroup=2, opacity=0.5)
        ],
        layout={
            'yaxis': {'title': 'SF Zoo axis'},
            'yaxis2': {'title': 'LA Zoo axis', 'overlaying': 'y', 'side': 'right'}
        }
    )

    # Update the layout of the figure
    fig.update_layout(
        title='Calibration Curve',
        xaxis_title='Predicted Probability',
        yaxis_title='True Probability in Each Bin',
        legend_x=0.5,
        legend_y=1.2,
        width=600,
        height=600
    )

    # Create the table figure
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=['Key', 'Value']),
        cells=dict(values=[['ECE'], [ece]])
    )])

    # Display the figures side by side
    fig_widget = go.FigureWidget(fig)
    table_fig_widget = go.FigureWidget(table_fig)
    return widgets.HBox([fig_widget, table_fig_widget])

def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    :param y_true: True binary labels
    :param y_pred: Predicted probabilities
    :param n_bins: Number of bins to use
    :return: Expected Calibration Error
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            avg_pred_prob = np.mean(y_pred[bin_mask])
            true_freq = np.mean(y_true[bin_mask])
            bin_error = np.abs(avg_pred_prob - true_freq)
            ece += (bin_size / len(y_true)) * bin_error

    return ece
