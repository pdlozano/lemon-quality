import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Callable
from torchvision.transforms import ToTensor


def default_transform(image: Image) -> torch.Tensor:
    """
    Turns an image to a tensor. Does no other transformations.

    :param image: An image of type PIL.Image
    :return: A tensor representation of the image in CHW format
    """
    transform = ToTensor()

    return transform(image)


class CustomDataset(Dataset):
    """
    A custom dataset to work with dataframes of filenames.

    :param data: A pandas dataframe with columns 'filenames'
        (string) and 'category' (string).
    :param base_dir: A path to the base directory of where
        the files are located. The categories' folders must
        be directly in the base_dir which contains the images.
        For example: `base_dir/category/image.jpg`.
    :param transform: A callable function that transforms a
        PIL image into a torch tensor. The default is a direct
        transform from an image to a tensor.
    """

    def __init__(
            self,
            X: pd.Series,
            y: pd.Series,
            base_dir: Path,
            transform: Callable[[Image], torch.Tensor] = default_transform
    ):
        self.__data_X = X
        self.__data_y = y
        self.classes = y.unique()
        # This is to maintain order between different dataframes
        self.classes = sorted(self.classes)
        self.class_to_idx = { key: index for index, key in enumerate(self.classes) }

        self.base_dir = base_dir
        self.transform = transform

    def __len__(self) -> int:
        """
        Gets the length of the dataset

        :return: The number of items in the dataset
        """
        return len(self.__data_X)

    def __getitem__(self, idx: int) -> (torch.Tensor, int):
        """
        Gets an item in `idx`. To transform the image class (which will be in
        an integer) to a string representation, use the `CustomDataset.classes`
        attribute.

        :param idx: An integer from the dataset
        :return: A tensor representation of the image and the image class
        """
        filename = self.__data_X.iloc[idx]
        category = self.__data_y.iloc[idx]


        image = Image.open(self.base_dir / category / filename)
        image_class = category
        image_class = self.class_to_idx[image_class]

        return self.transform(image), image_class
