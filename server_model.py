import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from PIL import Image
from typing import List

REAL_LABELS = ['bad_quality', 'empty_background', 'good_quality']


class BestModel:
    def __init__(
            self,
            models_path: Path,
            labels: List[str] = REAL_LABELS,
            device: torch.device = torch.device("cpu"),
    ):
        """
        Instantiates the best model for use. Use `.to` to transfer
        it to a device and `.predict` to predict a single image.
        
        :param models_path: A `Path` containing where the model's
            path file is stored.
        :param labels: A list of real labels in the images. As
            the output of the model will be integers (e.g. 0, 1), to
            transform it in a more readable human format, input a list
            of labels to transform it. There is a default `REAL_LABELS`
        :param device: 
        """
        self.labels = labels
        self.device = device

        model_weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        self.transforms = model_weights.transforms()
        self.model = torchvision.models.mobilenet_v3_small(weights=model_weights)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=3, bias=True),
        )
        self.model = self.model.to(device=device)

        state_dict = torch.load(models_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __repr__(self):
        return self.model.__repr__()

    def to(self, device: torch.device):
        self.device = device
        self.model = self.model.to(device=device)
        return self

    def predict(self, image: Image.Image) -> str:
        self.model.eval()
        im = self.transforms(image).to(device=self.device)
        im = im.unsqueeze(dim=0)

        with torch.inference_mode():
            preds = self.model(im)
            preds = preds.argmax(dim=1)

        return self.labels[preds]
