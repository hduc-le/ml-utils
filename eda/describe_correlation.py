import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score


def plot_confusion_matrix_with_details(y_true, y_pred, labels, figsize=(15, 15)):
    """
    Plots the confusion matrix with counts, percentages, and precision values.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    labels (list): List of labels to index the matrix.
    figsize (tuple): Figure size, default is (15, 15).
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate precision for each class
    precision = precision_score(y_true, y_pred, labels=labels, average=None)

    # Calculate the percentage for each cell
    cm_percentage = cm / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot confusion matrix with custom background and text color
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    cax = ax.matshow(cm, cmap="Blues")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, color="black")
    ax.set_yticklabels(labels, color="black")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate the confusion matrix with precision and percentage values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm_percentage[i, j]
            count = cm[i, j]
            color = "black" if i == j else "red"
            ax.text(
                j,
                i,
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                color=color,
            )

    # Annotate the plot with precision values on the right side of the matrix
    for i in range(cm.shape[0]):
        ax.text(
            cm.shape[1], i, f"Precision: {precision[i]:.2f}", va="center", color="black"
        )

    plt.title("Confusion Matrix", color="black")
    plt.xlabel("Predicted label", color="black")
    plt.ylabel("True label", color="black")

    plt.show()
