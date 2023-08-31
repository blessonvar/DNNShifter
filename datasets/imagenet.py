# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
import numpy as np
import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform


def _get_samples(root, y_name, y_num):
    y_dir = os.path.join(root, y_name)
    if not get_platform().isdir(y_dir): return []

    output = []

    for f in get_platform().listdir(y_dir):
        if get_platform().isdir(os.path.join(y_dir, f)):
            output += _get_samples(y_dir, f, y_num)
        elif f.lower().endswith('jpeg'):
            output.append((os.path.join(y_dir, f), y_num))

    return output


class Dataset(base.ImageDataset):
    """ImageNet"""

    def __init__(self, loc: str, image_transforms, enumerate_examples=False):
        # Load the data.
        classes = sorted(get_platform().listdir(loc))
        samples = []

        if get_platform().num_workers > 0:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=get_platform().num_workers)
            futures = [executor.submit(_get_samples, loc, y_name, y_num) for y_num, y_name in enumerate(classes)]
            for d in concurrent.futures.wait(futures)[0]: samples += d.result()
        else:
            for y_num, y_name in enumerate(classes):
                samples += _get_samples(loc, y_name, y_num)
        examples, labels = zip(*samples)

        super(Dataset, self).__init__(
            np.array(examples), np.array(labels), image_transforms,
            [self._normalization_transform()],
            enumerate_examples=enumerate_examples)

    @staticmethod
    def num_train_examples(): return 1281167

    @staticmethod
    def num_test_examples(): return 50000

    @staticmethod
    def num_classes(): return 1000

    @staticmethod
    def _augment_transforms():
        return [
            torchvision.transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(0.8, 1.25)),
            torchvision.transforms.RandomHorizontalFlip()
        ]

    @staticmethod
    def _transforms():
        return [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)]

    @staticmethod
    def _normalization_transform():
        return torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @classmethod
    def get_train_set(cls, use_augmentation, enumerate_examples=False):
        transforms = cls._augment_transforms() if use_augmentation else cls._transforms()
        return cls(os.path.join(cls.root(), 'train'), transforms, enumerate_examples)

    @classmethod
    def get_test_set(cls, enumerate_examples=False):
        return cls(os.path.join(cls.root(), 'val'), cls._transforms(), enumerate_examples)

    @staticmethod
    def example_to_image(example):
        with get_platform().open(example, 'rb') as fp:
            return Image.open(fp).convert('RGB')

    @staticmethod
    def root():
        return get_platform().imagenet_root

DataLoader = base.DataLoader
