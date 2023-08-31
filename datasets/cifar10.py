# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import pickle
from PIL import Image
import sys
import tempfile
import torchvision

from datasets import base
from platforms.platform import get_platform


class CIFAR10(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if self._check_integrity(): return

        if get_platform().is_primary_process:
            if not get_platform().exists(self.root):
                temporary_root = tempfile.mkdtemp()
                torchvision.datasets.utils.download_and_extract_archive(
                    self.url, temporary_root, filename=self.filename, md5=self.tgz_md5)
                get_platform().copytree(temporary_root, self.root)

    def _check_integrity(self):
        return all([get_platform().exists(os.path.join(self.root, self.base_folder, filename))
                    for filename, _ in self.train_list + self.test_list])

    def __init__(self, root, train=True, download=False):
        super(torchvision.datasets.CIFAR10, self).__init__(root, transform=None, target_transform=None)

        if download: self.download()
        if not self._check_integrity(): raise ValueError('Dataset not found.')

        self.train = train
        downloaded_list = self.train_list if train else self.test_list

        self.data, self.targets = [], []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with get_platform().open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


class Dataset(base.ImageDataset):
    """The CIFAR-10 dataset."""

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation, enumerate_examples=False):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [],
                       enumerate_examples=enumerate_examples)

    @staticmethod
    def get_test_set(enumerate_examples=False):
        test_set = CIFAR10(train=False, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        return Dataset(test_set.data, np.array(test_set.targets), enumerate_examples=enumerate_examples)

    def __init__(self,  examples, labels, image_transforms=None, enumerate_examples=False):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                                      enumerate_examples=enumerate_examples)

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
