# 2dglasses

## What is this?
Training a convolutional neural net to distinguish binary classes (intended usage: anime-style girls (2d) from photos of real life girls (3d)), and using deepdream methods to convert between images of those types.

Using modified version of CaffeNet à la http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html

The dataset I used is self-collected with duplicates removed, consisting of 9000 2d images and 3000 3d images, at a 80-20 training/validation split. Dataset is not available for distribution due to copyright. Trained network model available by request under fair use.

You will need: python, caffe (already compiled), imagemagick (convert utility), 2d and 3d images

## How to train from scratch
Error rate is 5.47% after 10000 iterations.

1. `export CAFFE=/path/to/caffe`
2. `python scripts/prepare_data.py /path/to/2d /path/to/3d` 
3. `./scripts/create_lmdb.sh`
4. `./scripts/make_mean.sh`
5. `$CAFFE/build/tools/caffe train --solver=models/scratch/solver.prototxt`

## How to bootstrap training with caffenet
Error rate is 1.36% after 10000 iterations with `bvlc_reference_caffenet`.

Instead of the last step above:

1. Download the caffenet model by running `./scripts/download_model_binary.py models/<model>`, where `<model>` is either `bvlc_reference_caffenet` or `finetune_flickr_style`. Copy the resulting `.caffemodel` file from the caffe directory to `./snapshots`.
2. `$CAFFE/build/tools/caffe train --solver=models/bootstrap/solver.prototxt --weights=snapshots/<model>.caffemodel`

## How to use the classifier

1. `export CAFFE=/path/to/caffe`
2. `./tools/classify.py <image>..`

## How to use the converter
It doesn't really work. May need some more tuning or reworking.

1. `export CAFFE=/path/to/caffe`
2. `./tools/converter <2d or 3d> <source> <destination>`

