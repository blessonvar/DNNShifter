# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import argparse
from dataclasses import dataclass, field, make_dataclass
import inspect

from cli import shared_args
import foundations.desc
from foundations.hparams import Hparams
from foundations.runner import Runner


@dataclass
class Branch(Runner):
    replicate: int
    branch_desc: foundations.desc.Desc

    # Interface that needs to be overriden for each branch.
    @staticmethod
    @abc.abstractmethod
    def DescType() -> type:
        """The description type for this branch. Override this."""
        pass

    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this branch. Override this."""
        pass

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of this branch. Override this."""
        pass

    @abc.abstractmethod
    def branch_function(self) -> None:
        """The method that is called to execute the branch.

        Override this method with any additional arguments that the branch will need.
        These arguments will be converted into command-line arguments for the branch.
        Each argument MUST have a type annotation. The first argument must still be self.
        """
        pass

    # Interface that is useful for writing branches.
    @property
    def desc(self) -> foundations.desc.Desc:
        """The subcommand-specific description of this experiment."""

        return self.branch_desc.desc

    @property
    def experiment_name(self) -> str:
        """The name of this experiment."""

        return self.branch_desc.hashname

    # Interface that deals with command line arguments.

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        defaults = shared_args.maybe_get_default_hparams()
        shared_args.JobArgs.add_args(parser)
        cls.BranchDesc.add_args(parser, defaults)

    @classmethod
    def create_from_args(cls, args: argparse.Namespace):
        return cls(args.replicate, cls.BranchDesc.create_from_args(args))

    @classmethod
    def create_from_hparams(cls, replicate, desc: foundations.desc.Desc, hparams: Hparams):
        return cls(replicate, cls.BranchDesc(desc, hparams))

    @abc.abstractmethod
    def run(self):
        pass

    # Initialize instances and subclasses (metaprogramming).
    def __init_subclass__(cls):
        """Metaprogramming: modify the attributes of the subclass based on information in run().

        The goal is to make it possible for users to simply write a single run() method and have
        as much functionality as possible occur automatically. Specifically, this function converts
        the annotations and defaults in run() into a `BranchHparams` property.
        """

        if cls.name() is None: return

        fields = []
        for arg_name, parameter in list(inspect.signature(cls.branch_function).parameters.items())[1:]:
            t = parameter.annotation
            if t == inspect._empty: raise ValueError(f'Argument {arg_name} needs a type annotation.')
            elif t in [str, float, int, bool] or (isinstance(t, type) and issubclass(t, Hparams)):
                if parameter.default != inspect._empty: fields.append((arg_name, t, field(default=parameter.default)))
                else: fields.append((arg_name, t))
            else:
                raise ValueError('Invalid branch type: {}'.format(parameter.annotation))

        fields += [('_name', str, 'Branch Arguments'), ('_description', str, 'Arguments specific to the branch.')]
        setattr(cls, 'BranchHparams', make_dataclass('BranchHparams', fields, bases=(Hparams,)))
        setattr(cls, 'BranchDesc', make_BranchDesc(cls.BranchHparams, cls.DescType(), cls.name()))


def make_BranchDesc(BranchHparams: type, UnderlyingDesc: type, name: str):
    @dataclass
    class BranchDesc(foundations.desc.Desc):
        desc: UnderlyingDesc
        branch_hparams: BranchHparams

        @staticmethod
        def name_prefix(): return f'{UnderlyingDesc.name_prefix()}_branch_' + name

        @staticmethod
        def add_args(parser: argparse.ArgumentParser, defaults: UnderlyingDesc = None):
            UnderlyingDesc.add_args(parser, defaults)
            BranchHparams.add_args(parser)

        @classmethod
        def create_from_args(cls, args: argparse.Namespace):
            return BranchDesc(UnderlyingDesc.create_from_args(args), BranchHparams.create_from_args(args))

    return BranchDesc
