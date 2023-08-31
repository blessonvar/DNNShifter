# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
from typing import List

from cli import shared_args
from foundations import paths
from foundations.step import Step
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform
from training import train
from training.desc import TrainingDesc


@dataclass
class TrainingRunner(Runner):
    replicate: int
    desc: TrainingDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True
    weight_save_steps: List[Step] = field(default_factory=list)

    @staticmethod
    def description():
        return "Train a model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        TrainingDesc.add_args(parser, shared_args.maybe_get_default_hparams())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        d = TrainingDesc.create_from_args(args)
        if args.weight_save_steps:
            weight_save_steps = [d.str_to_step(s) for s in args.weight_save_steps.split(',')]
        else:
            weight_save_steps = []
        return TrainingRunner(args.replicate, TrainingDesc.create_from_args(args),
                              not args.quiet, not args.evaluate_only_at_end, weight_save_steps)

    def display_output_location(self):
        print(self.desc.run_path(self.replicate))

    def run(self):
        if get_platform().exists(paths.model(self.desc.run_path(self.replicate), self.desc.end_step)): return
        if self.verbose and get_platform().is_primary_process:
            print('='*82 + f'\nTraining a Model (Replicate {self.replicate})\n' + '-'*82)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.replicate)}' + '\n' + '='*82 + '\n')
        self.desc.save(self.desc.run_path(self.replicate))
        train.standard_train(
            models.registry.get(self.desc.model_hparams), self.desc.run_path(self.replicate),
            self.desc.dataset_hparams, self.desc.training_hparams,
            evaluate_every_epoch=self.evaluate_every_epoch, verbose=self.verbose,
            weight_save_steps=self.weight_save_steps)
