# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import tempfile
import torchvision

from datasets import base
from platforms.platform import get_platform


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(torchvision.datasets.MNIST, self).__init__(
            root, transform=transform, target_transform=target_transform)
        self.download()

        self.train = train
        data_file = self.training_file if self.train else self.test_file
        self.data, self.targets = get_platform().load_model(os.path.join(self.processed_folder, data_file))

class Dataset(base.ImageDataset):
    """The MNIST dataset."""

    @staticmethod
    def num_train_examples(): return 60000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation, enumerate_examples=False):
        # No augmentation for MNIST.
        train_set = MNIST(
            train=True, root=os.path.join(get_platform().dataset_root, 'mnist'))
        return Dataset(train_set.data, train_set.targets, enumerate_examples)

    @staticmethod
    def get_test_set(enumerate_examples=False):
        test_set = MNIST(
            train=False, root=os.path.join(get_platform().dataset_root, 'mnist'))
        return Dataset(test_set.data, test_set.targets, enumerate_examples)

    def __init__(self,  examples, labels, enumerate_examples=False):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms, enumerate_examples=enumerate_examples)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')


DataLoader = base.DataLoader
