# **DNNShifter**: An Efficient DNN Pruning System for Edge Computing

## Environment Setup:
Follow these repo instructions to set up your environment, paths, datasets, number of works, etc.: \
https://github.com/facebookresearch/open_lth \
https://github.com/sahibsin/Pruning

## Training Sparse Models:
Run `dnnshifter-training.sh` to train a portfolio of sparse models, then run `othermethods-training-cifar10.sh`/`othermethods-training-tinyimagenet.sh` to train the other methods (Random, Magnitude, Synflow etc.) \
or \
Download pretained sparse models [here](https://universityofstandrews907-my.sharepoint.com/:f:/g/personal/bje1_st-andrews_ac_uk/EukhECFFbc5KkdPldcb-WgwBbdUSoatvV8epkLjs5lc94Q?e=Dqdn6G)

## DNNShifter Pruning, API, runtime performance, and post-pruning accuracy validation
Follow this [notebook](https://github.com/blessonvar/DNNShifter/blob/main/DNNShifter/notebook.ipynb) on how to use DNNShifter to prune sparse models structurally. \
(Note: Update paths where prompted, and models need to be stored in this directory structure format `<method_name>/<pruning_level>/<.pth file>`)
