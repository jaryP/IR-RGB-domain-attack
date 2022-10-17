#!/usr/bin/env bash

DEVICE=$1

python main.py +dataset=cifar10 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./results/training/ablation/cifar10/resnet20' +attacks=cifar10_search training.device="$DEVICE"
