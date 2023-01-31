import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(
        model: nn.Module,
        dataloader: DataLoader,
        labels: List[str],
        device: torch.device = torch.device("cpu"),
) -> ConfusionMatrixDisplay:
    """
    A function to plot the confusion matrix of predictions in a model

    :param model: The model whose predictions will be plotted
    :param dataloader: A Dataloader (using the `CustomDataset`)
        which contains images to be predicted by the model.
    :param labels: A list of real labels in the images. As
        the output of the model will be integers (e.g. 0, 1), to
        transform it in a more readable human format, input a list
        of labels to transform it.
    :param device: The device to use for model predictions
    :return: A `ConfusionMatrixDisplay` containing the model
        predictions. To plot it inline in Jupyter Notebook, use
        `result.plot()`.
    """
    y_true, y_pred = [], []

    model.to(device=device)
    model.eval()

    for image, label in dataloader:
        image, label = image.to(device=device), label.to(device=device)

        with torch.inference_mode():
            preds_logits = model(image)
            preds = preds_logits.argmax(dim=1)

            y_pred = [*y_pred, *preds.cpu()]
            y_true = [*y_true, *label.cpu()]

    y_pred = [labels[pred] for pred in y_pred]
    y_true = [labels[true] for true in y_true]

    return ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
        ),
        display_labels=labels,
    )
