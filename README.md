# 2dglasses

## What is this?
Currently, files used to train a convolutional neural net to distinguish binary classes (intended usage: anime-style girls (2d) from photos of real life girls (3d)).

Using modified version of CaffeNet à la http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html

The dataset I used is self-collected, consisting of 8k 2d images and 3k 3d images, at a 80%/20% training/validation split. Dataset is not available for distribution due to copyright. Trained network model available by request under fair use.

You will need: python, caffe (already compiled), imagemagick (convert utility), 2d and 3d images

## How to train from scratch
Error rate is 5.47% after 10000 iterations.

1. `export CAFFE=/path/to/caffe`
2. `python scripts/prepare_data.py /path/to/2d /path/to/3d` 
3. `./scripts/create_lmdb.sh`
4. `./scripts/make_mean.sh`
5. `./scripts/train_caffenet_scratch.sh`

## How to bootstrap training with caffenet
Error rate is 1.36% after 10000 iterations.

Instead of the last step above:

1. Download the caffenet model by running `./scripts/download_model_binary.py models/bvlc_reference_caffenet` in the caffe directory, and copy it to `./model`.
2. `./scripts/train_caffenet_bootstrap.sh`

