import torch

from training.branch.oneshot_experiments_helpers.base import PruningStrategy


class RandomPruning(PruningStrategy):
    @staticmethod
    def valid_name(strategy_name): return strategy_name == 'random'

    def score(self, model, mask):
        scores = {}
        for i, k in enumerate(sorted(model.prunable_layer_names)):
            generator = torch.Generator()
            generator.manual_seed(self._seed + i)
            scores[k] = torch.zeros_like(model.state_dict()[k].data.clone()).uniform_(0, 1, generator=generator)
        return scores

