from torch.utils.data import DataLoader
from scripts.customdataset import CustomDataset
from PIL import Image
from pathlib import Path
from typing import Callable
import pandas as pd
import torch


def make_dataloader(
        X: pd.Series,
        y: pd.Series,
        base_dir: Path,
        transform: Callable[[Image], torch.Tensor],
        shuffle: bool = False,
        batch_size: int = 32,
) -> DataLoader:
    """
    A function to automate the process of turning a set into a dataloader.

    :param set: A pandas dataframe with columns 'filenames'
        (string) and 'category' (string).
    :param base_dir: A path to the base directory of where
        the files are located. The categories' folders must
        be directly in the base_dir which contains the images.
        For example: `base_dir/category/image.jpg`.
    :param transform: A callable function that transforms a
        PIL image into a torch tensor. The default is a direct
        transform from an image to a tensor. Usually provided by
        the model you're using.
    :param shuffle: A boolean that tells whether the dataloader
        should shuffle the dataset. Default is `False`.
    :param batch_size: The batch size for the dataloader. Default
        is 32.
    :return: A torch.utils.data.DataLoader containing the data
    """

    dataset = CustomDataset(
        X=X, y=y,
        base_dir=base_dir,
        transform=transform,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
