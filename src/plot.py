"""
This module contains plotting utility functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_history(history: tf.keras.callbacks.History, model_name, fig_name=None):
    """
    Plot history of training and validation losses and accuracies.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title(f"{model_name} Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title(f"{model_name} BinaryCrossEntropyLoss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    
    if fig_name:
        plt.savefig(fig_name, dpi=150)
        
    plt.show()


def plot_confusion_matrix(cm, classes, fig_title, fig_name=None):
    """
    Plot dataset confusion matrix using ground truth labels and model predictions.
    """
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        cmap="Blues", xticks_rotation=45, ax=ax
    )
    ax.set_title(fig_title, fontsize=12)
    plt.tight_layout()
    
    if fig_name:
        plt.savefig(fig_name, dpi=150)
        
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, fig_title, fig_name=None):
    """
    Plot the ROC curve, the true positive rate against the false positive rate across all decision thresholds.
    """

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=fig_title).plot(ax=ax)
    ax.set_title(fig_title, fontsize=12)
    ax.legend()
    plt.tight_layout()
    
    if fig_name:
        plt.savefig(fig_name, dpi=150)

    plt.show()


def plot_precision_recall_curve(precisions, recalls, thresholds, fig_title, fig_name=None, threshold=None, bounds=[0.0, 1.0]):
    """
    Plot precision and recall across all decision thresholds.

    Optional arguments: 
    - threshold defines a decision boundary to plot a dashed line.
    - bounds limit the range of decision thresholds across the x axis.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Precision recall curve against threshold
    ax.set_xlim(bounds) # Bound decision thresholds on x-axis
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    ax.plot(thresholds, recalls[:-1], color="orange", linestyle="-", label="Recall", linewidth=2)

    # Plot line at specific threshold if provided
    if threshold:
        ax.vlines(threshold, 0, 1.0, "k", "dotted", label=f"Threshold {threshold}")
    
    ax.set_xlabel("Threshold")
    ax.set_title(fig_title)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    
    if fig_name:
        plt.savefig(fig_name, dpi=150)
    
    plt.show()


def plot_classification_report(cr, fig_title, fig_name=None):
    """
    Plot the classification report as a pyplot table.
    """
    cols = ["precision", "recall", "f1-score"]
    rows = [k for k in cr.keys() if isinstance(cr[k], dict)]
    data = [[round(cr[row][col], 3) for col in cols] for row in rows]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.table(
        cellText=data, 
        rowLabels=rows, 
        colLabels=cols,
        cellLoc="center",
        loc="center",
        colColours=["#d0e4f7"] * len(cols),
        rowColours=["#d0e4f7"] * len(rows),
    )
    ax.set_title(fig_title)
    plt.tight_layout()

    if fig_name:
        plt.savefig(fig_name, dpi=150)
        
    plt.show()