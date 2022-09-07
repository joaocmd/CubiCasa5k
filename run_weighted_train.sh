#!/bin/bash

python ./train.py --arch deeplabv3 --batch-size 16 --loss weighted
python ./train.py --arch dfp --batch-size 16 --loss weighted
python ./train.py --arch dfp-last-conv --batch-size 16 --loss weighted
python ./train.py --arch dfp-resnet34  --batch-size 16 --loss weighted
python ./train.py --arch dfp-resnet34-encoder-conv  --batch-size 16 --loss weighted
python ./train.py --arch dfp-resnet50  --batch-size 4 --loss weighted
