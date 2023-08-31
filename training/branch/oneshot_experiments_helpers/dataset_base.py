import abc
import collections
import copy
import torch
from typing import List

import datasets.registry
from foundations.hparams import Hparams
from models.base import Model
from pruning.mask import Mask
from training.branch.oneshot_experiments_helpers import base
from training.desc import TrainingDesc


Scores = dict


class PruningStrategy(base.PruningStrategy):
    def __init__(self, strategy_name: str, desc: TrainingDesc, seed: int):
        super(PruningStrategy, self).__init__(strategy_name, desc, seed)

        # Determine the details of the dataset.
        num_per_class = self.num_samples()
        dataset_hparams = copy.deepcopy(desc.dataset_hparams)
        dataset_hparams.do_not_augment = True

        # Get the dataset itself.
        d = datasets.registry.get(dataset_hparams, train=True, force_sequential=True, enumerate_examples=True)
        d.shuffle(seed)
        included = torch.zeros(d.dataset.num_train_examples())

        # Determine which examples to include.
        results = collections.defaultdict(int)
        num_classes = datasets.registry.num_classes(dataset_hparams)
        count = 0

        for indices, (_, labels) in d:
            for index, label in zip(indices, labels):
                label = label.int().item()
                included[index] = int(results[label] < num_per_class)
                results[label] += 1
                if results[label] == num_per_class: num_classes -= 1
                if num_classes == 0: break
            if num_classes == 0: break

        # Create the dataset with those examples.
        if dataset_hparams.dataset_name == 'imagenet': dataset_hparams.batch_size = 256
        self._dataset = datasets.registry.get(dataset_hparams, train=True, mask=included.numpy(), force_sequential=True)
        self._dataset.shuffle(seed)


    @abc.abstractmethod
    def num_samples(self):
        raise ValueError('Unimplemented')
