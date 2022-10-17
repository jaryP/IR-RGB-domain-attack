#!/usr/bin/env bash

DEVICE=$1

python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext20/cifar10' params.batch_size=32 params.img_size=none model.name=resnet20 params_attack.name=white test_params_attack.name=white dataset.name=cifar10 train_params.device="$DEVICE"

python main_adv_training.py hydra.run.dir='./results/adv_training/cifar10/resnet20/cifar10_50_20' params.batch_size=32 +params.img_size=none model.name=resnet20 params_attack.name=white_pixle test_params_attack.name=white_pixle dataset.name=cifar10 train_params.device=4 train_params.pre_training_epochs=50 train_params.adv_training_epochs=20 attacks=cifar10
python main_adv_training.py hydra.run.dir='./results/adv_training/ir_nir/resnext50_32x4d/cifar10_50_20' params.batch_size=32 +params.img_size=none model.name=resnext50_32x4d params_attack.name=white_pixle test_params_attack.name=white_pixle dataset.name=ir_nir train_params.device=4 train_params.pre_training_epochs=50 train_params.adv_training_epochs=20 +attacks=imagenet

#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/black_rgb' +attack_type.name=black dataset.name=rgb_nir train_params.device="$DEVICE"
#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/black_ir' +attack_type.name=black dataset.name=ir_nir train_params.device="$DEVICE"
#
#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext50_32x4d/black_rgb' model.name=resnext50_32x4d +attack_type.name=black dataset.name=rgb_nir train_params.device="$DEVICE"
#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext50_32x4d/black_ir' model.name=resnext50_32x4d +attack_type.name=black dataset.name=ir_nir train_params.device="$DEVICE"


#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/black_rgb' +attack_type.name=black dataset.name=rgb_nir train_params.device="$DEVICE"
#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/black_ir' +attack_type.name=black dataset.name=ir_nir train_params.device="$DEVICE"

python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext50_32x4d/white_rgb1' model.name=resnext50_32x4d params_attack.name=white test_params_attack.name=white dataset.name=rgb_nir train_params.device="$DEVICE"
python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext50_32x4d/white_ir1' model.name=resnext50_32x4d params_attack.name=white test_params_attack.name=white dataset.name=ir_nir train_params.device="$DEVICE"

# python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/white' train_params.device="$DEVICE"
