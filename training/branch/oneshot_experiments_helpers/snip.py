import torch

from training.branch.oneshot_experiments_helpers.dataset_base import PruningStrategy


class SNIP(PruningStrategy):
    @staticmethod
    def valid_name(strategy_name): return strategy_name.startswith('snip') and strategy_name[len('snip'):].isdigit()

    def num_samples(self):
        return int(self._strategy_name[len('snip'):])

    def score(self, model, mask):
        model.train()

        # Preliminaries.
        total = torch.tensor(0.0)
        scores = {k: torch.zeros_like(v.data) for k, v in model.named_parameters() if k in mask}

        # Run SNIP.
        for i, (batch_examples, batch_labels) in enumerate(self._dataset):
            print('snip', i)
            model.zero_grad()
            loss = model.loss_criterion(model(batch_examples), batch_labels).backward()
            scores = {
                k: scores[k] + torch.abs(v.data.clone().detach() * v.grad.clone().detach()) * len(batch_labels)
                for k, v in model.named_parameters() if k in scores
            }
            total += torch.tensor(len(batch_labels))

        return {k: v / total.item() for k, v in scores.items()}
