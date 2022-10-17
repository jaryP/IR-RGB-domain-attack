#!/usr/bin/env bash

DEVICE=$1

python main_transfer_training.py hydra.run.dir='./results/transfer/nir/white/resnext50_32x4d' params.batch_size=16 +params.img_size=none model.name=resnext50_32x4d params_attack.name=white_pixle test_params_attack.name=white_pixle dataset.name=ir_nir train_params.device="$DEVICE" train_params.pre_training_epochs=100 +attacks=transfer params.test=0.2
python main_transfer_training.py hydra.run.dir='./results/transfer/nir/white/convnext_tiny' params.batch_size=16 +params.img_size=none model.name=convnext_tiny params_attack.name=white_pixle test_params_attack.name=white_pixle dataset.name=ir_nir train_params.device="$DEVICE" train_params.pre_training_epochs=100 +attacks=transfer params.test=0.2
