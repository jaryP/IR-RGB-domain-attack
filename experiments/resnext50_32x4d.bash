#!/usr/bin/env bash

DEVICE=$1

python main.py +dataset=cifar10 experiment=base +model=resnext50_32x4d optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./results/cifar10/resnext50_32x4d' +attacks=cifar10 training.device="$DEVICE"
python main.py +dataset=cifar10 experiment=base +model=convnext_tiny optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./results/cifar10/convnext_tiny' +attacks=cifar10 training.device="$DEVICE"

#python main.py +dataset=cifar10 experiment=base +model=resnext50_32x4d optimizer=sgd_momentum +training=cifar10 +training.resize_dimensions=256 hydra.run.dir='./results/cifar10/resnext50_32x4d' +attacks=cifar10 training.device="$DEVICE"
#python main.py +dataset=cifar10 experiment=base +model=convnext_tiny optimizer=sgd_momentum +training=cifar10 +training.resize_dimensions=256 hydra.run.dir='./results/cifar10/convnext_tiny' +attacks=cifar10 training.device="$DEVICE"
