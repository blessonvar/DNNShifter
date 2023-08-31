import scipy.stats
import torch

from training.branch.oneshot_experiments_helpers.base import PruningStrategy
from utils.tensor_utils import vectorize, unvectorize


class MagnitudePruning(PruningStrategy):
    @staticmethod
    def valid_name(strategy_name):
        return strategy_name == 'magnitude'

    def score(self, model, mask):
        return {k: torch.abs(model.state_dict()[k].data.clone()) for k in model.prunable_layer_names}

    