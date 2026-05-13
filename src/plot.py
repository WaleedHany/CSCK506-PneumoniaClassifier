import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_history(history: tf.keras.callbacks.History, model_name, fig_name):
    """
    Plot history of training and validation losses and accuracies.
    """
    val_loss = history.history["val_loss"]
    best_epoch = int(np.argmin(val_loss))

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
    plt.savefig(fig_name, dpi=150)
    plt.show()


def plot_confusion_matrix(cm, classes, model_name, fig_name):
    """
    Plot dataset confusion matrix using ground truth labels and model predictions.
    """
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        cmap="Blues", xticks_rotation=45, ax=ax
    )
    ax.set_title(f"{model_name} Confusion Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=150)
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, model_name, fig_name):
    """
    Plot the ROC curve, the true positive rate against the false positive rate across all decision thresholds.
    """

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=model_name).plot(ax=ax)
    ax.set_title(model_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=150)
    plt.show()
    