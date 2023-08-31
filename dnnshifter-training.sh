#!/bin/bash

#VGG-16
python3 open_lth.py lottery --default_hparams=cifar_vgg_16 --levels=25 --rewinding_steps=2000it

# ResNet-50
python3 open_lth.py lottery --default_hparams=tinyimagenet_resnet_50 --levels=25 --rewinding_steps=2000it