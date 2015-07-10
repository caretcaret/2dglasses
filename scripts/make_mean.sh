#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=model
DATA=data
TOOLS=$CAFFE/build/tools

$TOOLS/compute_image_mean $DATA/train_lmdb \
  $DATA/train_mean.binaryproto

$TOOLS/compute_image_mean $DATA/val_lmdb \
  $DATA/val_mean.binaryproto

echo "Done."
