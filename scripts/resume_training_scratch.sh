#!/usr/bin/env sh

$CAFFE/build/tools/caffe train \
    --solver=model/solver_scratch.prototxt \
    --snapshot=snapshots/scratch/caffenet_train_10000.solverstate
