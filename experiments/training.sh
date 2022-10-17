#!/usr/bin/env bash

DATASET=$1
MODEL=$2
DEVICE=$3

case $DATASET in
  cifar10)
    case $MODEL in
    resnext50_32x4d)
      python main.py +dataset=cifar10 experiment=base +training.resize_dimensions=128 +model=resnext50_32x4d optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./results/training/cifar10/resnext50_32x4d' +attacks=cifar10 training.device="$DEVICE"
    ;;
    convnext_tiny)
      python main.py +dataset=cifar10 experiment=base +training.resize_dimensions=128 +model=convnext_tiny optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./results/training/cifar10/convnext_tiny' +attacks=cifar10 training.device="$DEVICE"
    ;;
    resnet20)
      python main.py +dataset=cifar10 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./results/training/cifar10/resnet20' +attacks=cifar10 training.device="$DEVICE"
    ;;
    vgg11)
      python main.py +dataset=cifar10 experiment=base +model=vgg11 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./results/training/cifar10/vgg11' +attacks=cifar10 training.device="$DEVICE"
    ;;
    *)
      echo -n "Unrecognized model"
    esac
  ;;
  tinyimagenet)
    case $MODEL in
      resnext50_32x4d)
        python main.py +dataset=tinyimagenet experiment=base +training.resize_dimensions=128 +model=resnext50_32x4d optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./results/training/tinyimagenet/resnext50_32x4d' +attacks=imagenet training.device="$DEVICE"
      ;;
      convnext_tiny)
        python main.py +dataset=tinyimagenet experiment=base +training.resize_dimensions=128 +model=convnext_tiny optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./results/training/tinyimagenet/convnext_tiny/' +attacks=imagenet training.device="$DEVICE"
      ;;
      *)
      echo -n "Unrecognized model"
    esac
  ;;
  imagenet)
    case $MODEL in
      resnext50_32x4d)
        python main.py +dataset=imagenet experiment=base +model=resnext50_32x4d +model.pretrained=true optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./results/training/imagenet/resnext50_32x4d/' +attacks=imagenet training.device="$DEVICE"
      ;;
      convnext_tiny)
        python main.py +dataset=imagenet experiment=base +model=convnext_tiny +model.pretrained=true optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./results/training/imagenet/convnext_tiny/' +attacks=imagenet training.device="$DEVICE"
      ;;
      *)
      echo -n "Unrecognized model"
    esac
  ;;
  ir_nir)
    case $MODEL in
      resnext50_32x4d)
        python main.py +dataset=nir_ir experiment=base +model=resnext50_32x4d +model.pretrained=true optimizer=sgd_momentum +training=nir_ir hydra.run.dir='./results/training/ir_nir/resnext50_32x4d/' +attacks=imagenet training.device="$DEVICE"
      ;;
      convnext_tiny)
        python main.py +dataset=nir_ir experiment=base +model=convnext_tiny +model.pretrained=true optimizer=sgd_momentum +training=nir_ir hydra.run.dir='./results/training/ir_nir/convnext_tiny/' +attacks=imagenet training.device="$DEVICE"
      ;;
      *)
      echo -n "Unrecognized model"
    esac
  ;;
  rgb_nir)
    case $MODEL in
      resnext50_32x4d)
        python main.py +dataset=nir_rgb experiment=base +model=resnext50_32x4d +model.pretrained=true optimizer=sgd_momentum +training=nir_rgb hydra.run.dir='./results/training/nir_rgb/resnext50_32x4d/' +attacks=imagenet training.device="$DEVICE"
      ;;
      convnext_tiny)
        python main.py +dataset=nir_rgb experiment=base +model=convnext_tiny +model.pretrained=true optimizer=sgd_momentum +training=nir_rgb hydra.run.dir='./results/training/nir_rgb/convnext_tiny/' +attacks=imagenet training.device="$DEVICE"
      ;;
      *)
      echo -n "Unrecognized model"
    esac
  ;;
  *)
  echo -n "Unrecognized dataset"
esac
