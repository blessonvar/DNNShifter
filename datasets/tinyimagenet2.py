import os
from PIL import Image
import numpy as np
import torchvision

from datasets import base
from datasets import tinyimagenet
from platforms.platform import get_platform

class Dataset(tinyimagenet.Dataset):
    @staticmethod
    def _augment_transforms():
        return [
            torchvision.transforms.RandomCrop(64, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ]

DataLoader = base.DataLoader
