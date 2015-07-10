#!/usr/bin/env sh

$CAFFE/build/tools/caffe train \
    --solver=model/solver.prototxt \
    --snapshot=snapshots/caffenet_train_10000.solverstate
