from torch.utils.data import DataLoader
from typing import Callable
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import torch.nn as nn
import torch

default_dataframe = pd.DataFrame({
    'epoch': [],
    'train-loss': [],
    'train-metrics': [],
    'validation-loss': [],
    'validation-metrics': [],
})


def train_model(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        metrics_fn: Callable[[torch.Tensor, torch.Tensor], float],
        device: torch.device,
) -> pd.Series:
    """
    A function to train the model for ONE step

    :param model: The model to be trained. Has to be nn.Module
    :param dataloader: The dataloader containing the data. It must
        use the `scripts.customdataset.CustomDataset` dataset.
    :param loss_fn: A loss function to be used to train the model
    :param optimizer: The optimizer used to train the model
    :param metrics_fn: A metrics function to evaluate the model's
        performance
    :param device: The device to be used to train the model
    :return: A Pandas Series containing loss and metrics
        for the single step
    """
    model.train()

    total_loss = 0
    total_metrics = 0

    for batch, (image, label) in enumerate(dataloader):
        image = image.to(device=device)
        label = label.to(device=device)

        preds_logits = model(image)
        preds = preds_logits.argmax(dim=1)

        loss = loss_fn(preds_logits, label)
        metrics = metrics_fn(label.cpu(), preds.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_metrics += metrics

        if batch % 10 == 0:
            print(f"Train Batch {batch} ({batch * len(image)} / {len(dataloader.dataset)})")

    batch_loss = total_loss.cpu().detach().numpy() / len(dataloader)
    batch_metrics = total_metrics / len(dataloader)

    return pd.Series({
        'train-loss': batch_loss,
        'train-metrics': batch_metrics,
    })

def validate_model(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer, # For consistency purposes
        metrics_fn: Callable[[torch.Tensor, torch.Tensor], float],
        device: torch.device,
) -> pd.Series:
    """
    A function to validate the model for ONE step. The data in the
    validation dataloader will not be used to train the model.

    :param model: The model to be validated. Has to be nn.Module
    :param dataloader: The dataloader containing the data. It must
        use the `scripts.customdataset.CustomDataset` dataset.
    :param loss_fn: A loss function to be used to validate the model
    :param optimizer: This is a parameter to be consistent with
        the `train_model` function. It does nothing to the
        validation function. If you don't have an optimizer, you can
        use `optimizer=None`.
    :param metrics_fn: A metrics function to evaluate the model's
        performance
    :param device: The device to be used to validate the model
    :return: A Pandas Series containing loss and metrics
        for the single step
    """
    model.eval()

    total_loss = 0
    total_metrics = 0

    for batch, (image, label) in enumerate(dataloader):
        with torch.inference_mode():
            image = image.to(device=device)
            label = label.to(device=device)

            preds_logits = model(image)
            preds = preds_logits.argmax(dim=1)

            loss = loss_fn(preds_logits, label)
            metrics = metrics_fn(label.cpu(), preds.cpu())

            total_loss += loss
            total_metrics += metrics

            if batch % 10 == 0:
                print(f"Validation Batch {batch} ({batch * len(image)} / {len(dataloader.dataset)})")

    batch_loss = total_loss.cpu().detach().numpy() / len(dataloader)
    batch_metrics = total_metrics / len(dataloader)

    return pd.Series({
        'validation-loss': batch_loss,
        'validation-metrics': batch_metrics,
    })

def train_and_validate_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        metrics_fn: Callable[[torch.Tensor, torch.Tensor], float],
        device: torch.device,
        epochs: int,
        save_path: Path,
        previous: pd.DataFrame = default_dataframe,
) -> pd.DataFrame:
    """
    A function to train and validate the model for n epochs.

    :param model: The model to be trained. Has to be nn.Module
    :param train_dataloader: The dataloader containing the data
        for training. It must use the
        `scripts.customdataset.CustomDataset` dataset.
    :param validation_dataloader: The dataloader containing the data
        for validation. It must use the
        `scripts.customdataset.CustomDataset` dataset.
    :param loss_fn: A loss function to be used for both
        training and validation
    :param optimizer: The optimizer used for training and validation
    :param metrics_fn: A metrics function to evaluate the model's
        performance
    :param device: The device to be used to train the model
    :param epochs: The number of epochs to run the model with
    :param save_path: A `Path` to save the model's results to
    :param previous: To continue training at a particular step, input
        the previous results of the model here.
    :return: A dataframe containing the training and validation
        losses and metrics for the whole training and validation phase
    """
    model_name = model.__class__.__name__
    save_path.mkdir(parents=True, exist_ok=True)
    model.to(device=device)
    metrics = previous.copy()

    start = len(previous)

    for epoch in tqdm(range(start, epochs)):
        train_metrics = train_model(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics_fn=metrics_fn,
            device=device,
        )

        validation_metrics = validate_model(
            model=model,
            dataloader=validation_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics_fn=metrics_fn,
            device=device,
        )

        epoch_metrics = pd.concat([
            pd.Series({ 'epoch': epoch }),
            train_metrics,
            validation_metrics,
        ])
        metrics = pd.concat([
            metrics,
            pd.DataFrame([epoch_metrics]),
        ])
        torch.save(model.state_dict(), save_path / f"{model_name}-{epoch + 1}.pt")

        print(f"Epoch {epoch + 1} / {epochs}")
        print(epoch_metrics)
        print()

    return metrics
