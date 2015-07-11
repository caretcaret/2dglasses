#!/usr/bin/env sh

$CAFFE/build/tools/caffe train \
    --solver=model/solver_bootstrap.prototxt \
    --snapshot=snapshots/bootstrap/caffenet_train_10000.solverstate
