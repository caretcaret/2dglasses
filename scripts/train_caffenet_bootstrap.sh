#!/usr/bin/env sh

$CAFFE/build/tools/caffe train \
    --solver=model/solver_bootstrap.prototxt \
    --weights=model/bvlc_reference_caffenet.caffemodel
