import copy
import torch

from training.branch.oneshot_experiments_helpers.dataset_base import PruningStrategy


class SynFlow(PruningStrategy):
    @staticmethod
    def valid_name(strategy_name): return strategy_name == 'synflow'

    def num_samples(self): return 1

    def score(self, model, mask):
        # Prepare the model.
        model = model.double()
        with torch.no_grad():
            for k, v in model.named_parameters(): v.abs_()
        model.eval()

        # Create an input with all ones.
        examples = next(iter(self._dataset))[0]
        input_dim = list(examples[0,:].shape)
        input = torch.ones([1] + input_dim).double()

        # Issue scores.
        torch.sum(model(input)).backward()
        return {k: torch.clone(v.grad*v).detach().abs_() for k, v in model.named_parameters() if k in mask}
