#!/bin/bash

set -e

echo "Training w/ scale"
python ./train.py --device cuda:3 --min-l-rate -1 --stopping-patience 1000 --scale --arch deeplabv3 --batch-size 16 --loss weighted --n-epoch 100

# TODO
# - Rename folder w/ resulting model to "runs_cubi/weighted-deeplabv3-with-scale"
# echo "Resume from scale"
# python ./train.py --device cuda:3 --min-l-rate -1 --stopping-patience 1000 --scale --arch deeplabv3 --batch-size 16 --loss weighted --resume-epoch --n-epoch 500 --weights "runs_cubi/weighted-deeplabv3-with-scale"
