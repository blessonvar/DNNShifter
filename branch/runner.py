# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
import sys

from cli import arg_utils
from foundations.runner import Runner
from branch import registry


@dataclass
class BranchRunner(Runner):
    """A meta-runner that calls the branch-specific runner."""

    runner: Runner

    @staticmethod
    def description():
        return "Run a branch."

    @staticmethod
    def add_args(parser):
        # Produce help text for selecting the branch.
        helptext = '='*82 + '\nOpenLTH: A Library for Research on Lottery Tickets and Beyond\n' + '-'*82
        runner_name = arg_utils.maybe_get_arg('runner', positional=True, position=1)

        # If the runner name is not present.
        if runner_name is None or runner_name not in registry.registered_runners():
            helptext = '\nChoose a runner on which to branch:\n'
            helptext += '\n'.join([f'    * {sys.argv[0]} branch {runner}' for runner in registry.registered_runners()])
            helptext += '\n' + '='*82
            print(helptext)
            sys.exit(1)

        # If the branch name is not present.
        branch_names = registry.registered_branches(runner_name)
        branch_name = arg_utils.maybe_get_arg('branch', positional=True, position=2)
        if branch_name is None or branch_name not in branch_names:
            helptext += '\nChoose a branch to run:'
            for bn in branch_names:
                helptext += "\n    * {} {} {} [...] => {}".format(
                            sys.argv[0], sys.argv[1], bn,
                            registry.get(runner_name, bn).description())
            helptext += '\n' + '='*82
            print(helptext)
            sys.exit(1)

        # Add the arguments for the branch.
        parser.add_argument('runner_name', type=str)
        parser.add_argument('branch_name', type=str)
        registry.get(runner_name, branch_name).add_args(parser)

    @staticmethod
    def create_from_args(args: argparse.Namespace):
        runner_name = arg_utils.maybe_get_arg('runner', positional=True, position=1)
        branch_name = arg_utils.maybe_get_arg('branch', positional=True, position=2)
        return BranchRunner(registry.get(runner_name, branch_name).create_from_args(args))

    def display_output_location(self):
        self.runner.display_output_location()

    def run(self) -> None:
        self.runner.run()


class LotteryBranch(BranchRunner):
    @staticmethod
    def description():
        return "Run a lottery branch."
