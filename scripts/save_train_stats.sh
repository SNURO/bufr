#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 1

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 2

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 3

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 4

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 5      

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 6

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 7

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 8

python save_train_stats.py    --data-root /gallery_tate/wonjae.roh/ \
                              --output-dir ./ \
                              --alg-config ./configs/CIFAR-10-C/save_train_stats.yml \
                              --data-config ./configs/CIFAR-10-C/dataset.yml \
                              --seed 123 \
                              --deterministic \
                              --n-workers 4 \
                              --pin-mem \
                              --class-idx 9                  
                        
