#!/usr/bin/env bash

DEVICE=$1

#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/black_rgb' +attack_type.name=black dataset.name=rgb_nir train_params.device="$DEVICE"
#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/black_ir' +attack_type.name=black dataset.name=ir_nir train_params.device="$DEVICE"
#
#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext50_32x4d/black_rgb' model.name=resnext50_32x4d +attack_type.name=black dataset.name=rgb_nir train_params.device="$DEVICE"
#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext50_32x4d/black_ir' model.name=resnext50_32x4d +attack_type.name=black dataset.name=ir_nir train_params.device="$DEVICE"


#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/black_rgb' +attack_type.name=black dataset.name=rgb_nir train_params.device="$DEVICE"
#python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/black_ir' +attack_type.name=black dataset.name=ir_nir train_params.device="$DEVICE"

python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext50_32x4d/white_rgb' model.name=resnext50_32x4d params_attack.name=white test_params_attack.name=white dataset.name=rgb_nir train_params.device="$DEVICE"
python main_adv_training.py hydra.run.dir='./results/adv_training/nir/resnext50_32x4d/white_ir' model.name=resnext50_32x4d params_attack.name=white test_params_attack.name=white dataset.name=ir_nir train_params.device="$DEVICE"

# python main_adv_training.py hydra.run.dir='./results/adv_training/nir/convnext_tiny/white' train_params.device="$DEVICE"
