#!/usr/bin/env sh

$CAFFE/build/tools/caffe train \
    --solver=model/solver_scratch.prototxt
