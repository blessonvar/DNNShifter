import os
from PIL import Image
import numpy as np
import torchvision

from datasets import base
from datasets import imagenet
from platforms.platform import get_platform

class Dataset(imagenet.Dataset):
    """
    Tiny ImageNet
    Download zip in directory of choice:
        wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    Unzip:
        unzip tiny-imagenet-200.zip
    """

    def __init__(self, loc:str, image_transforms, enumerate_examples=False):
        if loc.endswith('train'):
            super(Dataset, self).__init__(loc, image_transforms, enumerate_examples)
            return

        # Test and validation sets have an annotation file.
        annotations_file = os.path.join(loc, f'{os.path.basename(loc)}_annotations.txt')
        with get_platform().open(annotations_file) as fp:
            annotations = fp.read().split('\n')
        annotations = [(annotation.split()[0], annotation.split()[1]) for annotation in annotations if annotation.strip()]

        classes = sorted(list(set([c for _, c in annotations])))
        labels_dict = {c: i for i, c in enumerate(classes)}
        examples, labels = zip(*[(os.path.join(loc, 'images', f), labels_dict[c]) for f, c in annotations])

        super(imagenet.Dataset, self).__init__(
            np.array(examples), np.array(labels), image_transforms,
            [self._normalization_transform()],
            enumerate_examples=enumerate_examples)

    @staticmethod
    def _normalization_transform():
        # Note: TinyImageNet appears to already be normalized.
        return torchvision.transforms.Normalize([-0.021, -0.0354, -0.0376], [1.21, 1.20, 1.25])

    @staticmethod
    def num_train_examples(): return 100000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 200

    @staticmethod
    def _augment_transforms():
        return [
            torchvision.transforms.RandomResizedCrop(64, scale=(0.1, 1.0), ratio=(0.8, 1.25)),
            #torchvision.transforms.Resize(64),
            torchvision.transforms.RandomHorizontalFlip()
        ]

    @staticmethod
    def _transforms():
        return [torchvision.transforms.Resize(64)]

    @staticmethod
    def root(): return get_platform().tinyimagenet_root

DataLoader = base.DataLoader
