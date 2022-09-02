#!/usr/bin/env bash

DEVICE=$1

declare -a arr=("cifar10" "tinyimagenet" "imagenet")
declare -a arr1=("resnext50_32x4d" "convnext_tiny")

for i in "${arr[@]}"
do
  for j in "${arr1[@]}"
  do
    bash experiments/training.sh "$i" "$j" "$DEVICE"
  done
done
