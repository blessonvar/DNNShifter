# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from branch.branch import Branch
from training.branch import oneshot_experiments

_registry = {
    'train': {
        'oneshot': oneshot_experiments.Branch,
    },
}


def register(runner, name, branch):
    _registry[runner][name] = branch


def registered_runners():
    return _registry.keys()


def registered_branches(runner):
    return _registry[runner].keys()


def get(runner: str, branch_name: str) -> Branch:
    if runner not in _registry:
        raise ValueError(f'No such runner: {runner}')
    if branch_name not in _registry[runner]:
        raise ValueError(f'No such branch: {branch_name}')
    else:
        return _registry[runner][branch_name]
