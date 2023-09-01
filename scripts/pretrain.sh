#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

python pretrain.py  --data-root /gallery_tate/wonjae.roh/ \
                    --output-dir ./ \
                    --alg-config ./configs/CIFAR-10-C/pretrain.yml \
                    --data-config ./configs/CIFAR-10-C/dataset.yml \
                    --seed 123 \
                    --test-accuracy \
                    --deterministic \
                    --n-workers 4 \
                    --pin-mem

