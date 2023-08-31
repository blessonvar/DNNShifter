import abc
import collections
import copy
import torch

import datasets.registry
from foundations.hparams import Hparams
from models.base import Model
from pruning.mask import Mask
from training.desc import TrainingDesc


Scores = dict


class PruningStrategy(abc.ABC):
    def __init__(self, strategy_name: str, desc: TrainingDesc, seed: int):
        if not self.valid_name(strategy_name): raise ValueError(f'Invalid name: {strategy_name}')
        self._strategy_name = strategy_name
        self._desc = desc
        self._seed = seed

    @staticmethod
    @abc.abstractmethod
    def valid_name(strategy_name: str): raise ValueError('Unimplemented')

    @abc.abstractmethod
    def score(self, models: Model, mask: Mask) -> Scores:
        raise ValueError('Unimplemented')
