# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, fields, field
from typing import List

from branch.branch import Branch
from foundations.hparams import Hparams
from foundations.step import Step
from training.desc import TrainingDesc
from platforms.platform import get_platform


@dataclass
class TrainingBranch(Branch):
    """A training branch. Implement `branch_function`, add a name and description, and add to the registry."""

    evaluate_every_epoch: bool = True
    verbose: bool = False
    weight_save_steps: List[Step] = field(default_factory=list)

    @staticmethod
    def DescType():
        return TrainingDesc

    # Interface that is useful for writing branches.
    @property
    def branch_root(self) -> str:
        """The root for where branch results will be stored for a specific invocation of run()."""

        return self.desc.run_path(self.replicate, self.experiment_name)

    @classmethod
    def create_from_args(cls, args: argparse.Namespace):
        d = cls.DescType().create_from_args(args)
        if args.weight_save_steps:
            weight_save_steps = [d.str_to_step(s) for s in args.weight_save_steps.split(',')]
        else:
            weight_save_steps = []

        return cls(args.replicate, cls.BranchDesc.create_from_args(args),
                   not args.evaluate_only_at_end, not args.quiet, weight_save_steps)

    @classmethod
    def create_from_hparams(cls, replicate, desc: TrainingDesc, hparams: Hparams, verbose=False):
        return cls(replicate, cls.BranchDesc(desc, hparams), verbose)

    def display_output_location(self):
        print(self.branch_root)

    def run(self):
        if self.verbose and get_platform().is_primary_process:
            print('='*82)
            print(f'Branch {self.name()} (Replicate {self.replicate})\n' + '-'*82)
            print(f'{self.desc.display}\n{self.branch_desc.branch_hparams.display}')
            print(f'Output Location: {self.branch_root}\n' + '='*82 + '\n')

        args = {f.name: getattr(self.branch_desc.branch_hparams, f.name)
                for f in fields(self.BranchHparams) if not f.name.startswith('_')}
        self.branch_function(**args)
