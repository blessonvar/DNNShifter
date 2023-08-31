#!/bin/bash

#Random
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.0
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.5
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.75
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.875
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.9375
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.96875
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.98438
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.99219
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.99609

#LTH
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.0
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.5
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.75
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.875
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.9375
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.96875
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.98438
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.99219
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.99609

#Synflow
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.0 --prune_iterations=100
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.5 --prune_iterations=100
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.75 --prune_iterations=100
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.875 --prune_iterations=100
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.9375 --prune_iterations=100
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.96875 --prune_iterations=100
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.98438 --prune_iterations=100
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.99219 --prune_iterations=100
python3 open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.99609 --prune_iterations=100