# Importing libraries ...
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
import itertools


# ==== Plot loss and accuracy ====
def plotLossAndAccuracy(dataframe, fname, cut_epochs):
    """
    Plot loss and accuracy and save extract as svg format
    :param dataframe: experiment history dataframe
    :param fname: picture path
    :param cut_epochs: the epochs which finetuning occurs (0 if we do only feature extractor)
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(30, 12))
    num_epochs = len(dataframe)

    # Plot Loss result
    axs[0].plot(np.arange(num_epochs), dataframe['Train_loss'], label='Train loss', color='b')
    axs[0].plot(np.arange(num_epochs), dataframe['Val_loss'], label='Val loss', color='r')
    axs[0].set_xlabel("Epochs", fontsize=15)
    axs[0].set_ylabel("Loss", fontsize=15)
    axs[0].set_title("Loss Plot", fontsize=18)
    if cut_epochs > 0:
        axs[0].vlines(cut_epochs, 0, 1.6, linestyle='--', color='g')
    axs[0].legend(fontsize=14)
    axs[0].xaxis.set_tick_params(labelsize=14)
    axs[0].yaxis.set_tick_params(labelsize=14)

    # Plot Accuracy result
    axs[1].plot(np.arange(num_epochs), dataframe['Train_acc'], label='Train Accuracy', color='b')
    axs[1].plot(np.arange(num_epochs), dataframe['Val_acc'], label='Val Accuracy', color='r')
    axs[1].set_xlabel("Epochs", fontsize=15)
    axs[1].set_ylabel("Accuracy", fontsize=15)

    # Find the best model
    max_id = dataframe['Val_acc'].argmax()

    axs[1].set_title(
        f"Best Validation Accuracy: {dataframe['Val_acc'][max_id]:.2f}%\nBest Train Accuracy: {dataframe['Train_acc'][max_id]:.2f}%",
        fontsize=18)
    axs[1].plot(max_id, dataframe['Val_acc'][max_id], 'ro', markersize=12)
    axs[1].text(max_id, dataframe['Val_acc'][max_id] + 1, f'Best Model\nEpochs {max_id}', va='bottom', ha='center',
                fontsize=14)
    if cut_epochs > 0:
        axs[1].vlines(cut_epochs, 40, 100, linestyle='--', color='g')
    axs[1].legend(fontsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)
    axs[1].yaxis.set_tick_params(labelsize=14)

    # Save figure
    fig.savefig(fname, format='svg', dpi=1200)


# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.
    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).

    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.
    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")


# Function to evaluate: accuracy, precision, recall, f1-score
def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.
  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array
  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results