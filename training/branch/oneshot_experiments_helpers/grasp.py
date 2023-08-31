import copy
import numpy as np
import time
import torch

from training.branch.oneshot_experiments_helpers.dataset_base import PruningStrategy


class GraSP(PruningStrategy):
    @staticmethod
    def valid_name(strategy_name):
        if strategy_name.startswith('graspabs') and strategy_name[len('graspabs'):].isdigit():
            return True
        elif strategy_name.startswith('grasp') and strategy_name[len('grasp'):].isdigit():
            return True
        return False

    def num_samples(self):
        prefix_len = len('graspabs') if self._strategy_name.startswith('graspabs') else len('grasp')
        return int(self._strategy_name[prefix_len:])

    def score(self, model, mask):
        model.train()

        # Preliminaries.
        T = 200
        total = torch.tensor(0.0)
        scores = {k: torch.zeros_like(v.data) for k, v in model.named_parameters() if k in mask}

        # Run GraSP.
        t = None
        for i, (batch_examples, batch_labels) in enumerate(self._dataset):
            print('grasp', i, np.round(time.time() - t, 2) if t else '')
            t = time.time()
            model.zero_grad()

            weights = [v for k, v in model.named_parameters() if k in mask]
            loss = model.loss_criterion(model(batch_examples) / T, batch_labels)
            grad_w = list(torch.autograd.grad(loss, weights))

            loss = model.loss_criterion(model(batch_examples) / T, batch_labels)
            grad_f = list(torch.autograd.grad(loss, weights, create_graph=True))
            #print('grad_f')

            z = sum([(gw.data * gf).sum() for gw, gf in zip(grad_w, grad_f)])
            z.backward()
            #print('z')

            scores = {
                k: scores[k] + -v.data.clone().detach() * v.grad.clone().detach() * len(batch_labels)
                for k, v in model.named_parameters() if k in scores
            }
            total += torch.tensor(len(batch_labels))

        # Wrap up.
        if self._strategy_name.startswith('graspabs'): scores = {k: torch.abs(v) for k, v in scores.items()}
        return {k: v / total.item() for k, v in scores.items()}
