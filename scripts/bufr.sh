#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python bufr.py      --data-root /gallery_tate/wonjae.roh/ \
                    --output-dir ./ \
                    --alg-configs-dir ./configs/CIFAR-10-C/ \
                    --data-config ./configs/CIFAR-10-C/dataset.yml \
                    --seed 123 \
                    --deterministic \
                    --n-workers 4 \
                    --pin-mem \
                    --save-adapted-model