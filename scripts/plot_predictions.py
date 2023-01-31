from typing import List, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def create_normalizer(
        images: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a function to normalize the image between
    0 and 1

    :param images: A tensor of images to normalize
    :return: A function that takes in a tensor and normalizes
        its values between 0 and 1
    """
    min_val = images.min().cpu()
    max_val = images.max().cpu()

    def normalizer(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - min_val) / (max_val - min_val)

    return normalizer


def plot_model_predictions(
        model: nn.Module,
        dataloader: DataLoader,
        real_labels: List[str],
        device: torch.device = torch.device("cpu"),
) -> plt.Figure:
    """
    Plots the model predictions in a 4x8 grid.

    :param model: The model whose predictions will be plotted
    :param dataloader: A Dataloader (using the `CustomDataset`)
        which contains images to be predicted by the model.
    :param real_labels: A list of real labels in the images. As
        the output of the model will be integers (e.g. 0, 1), to
        transform it in a more readable human format, input a list
        of labels to transform it.
    :param device: The device to use for model predictions
    :return: A Figure containing the model predictions. To plot
        it inline in Jupyter Notebook, use `result.plot()`.
    """
    model.to(device=device)

    images, labels = next(iter(dataloader))
    images, labels = images.to(device=device), labels.to(device=device)
    normalizer = create_normalizer(images)

    fig = plt.figure(figsize=(10, 8))

    with torch.inference_mode():
        preds = model(images)
        preds = preds.argmax(dim=1)

    for index, image in enumerate(images):
        real = labels[index]
        pred = preds[index]
        text_color = "green" if real == pred else "red"
        im = normalizer(image.cpu())
        im = im.permute((1, 2, 0))

        plt.subplot(4, 8, index + 1)
        plt.imshow(im)
        plt.title(
            f"Real: {real_labels[real]}\nPred: {real_labels[pred]}",
            c=text_color,
            fontsize=6,
        )
        plt.axis(False)

    plt.close()  # Prevents IPYNB from displaying the plot twice
    return fig
