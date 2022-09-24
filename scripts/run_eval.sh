#!/bin/bash

set -e

echo "Running evaluation"
INPUT_DIR=runs_cubi/resume-from-scale
OUTPUT_DIR=output/results-resume-from-scale
MODEL_TYPE=model_best_val_loss_var.pkl
SPLIT_CONFIG="--split val.txt"
DEVICE_CONFIG="--device cuda:0"
echo $INPUT_DIR $OUTPUT_DIR

# echo "Eval Furukawa Original baseline"
python eval.py --arch hg_furukawa_original --weights "${INPUT_DIR}/../baseline/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/${INPUT_DIR}/../baseline" $SPLIT_CONFIG $DEVICE_CONFIG
    
python eval.py --arch "dfp" --weights "${INPUT_DIR}/dfp/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/${INPUT_DIR}/dfp" $SPLIT_CONFIG $DEVICE_CONFIG
python eval.py --arch "dfp-last-conv" --weights "${INPUT_DIR}/dfp-last-conv/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/${INPUT_DIR}/dfp-last-conv" $SPLIT_CONFIG $DEVICE_CONFIG
python eval.py --arch "dfp-resnet34" --weights "${INPUT_DIR}/dfp-resnet34/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/${INPUT_DIR}/dfp-resnet34" $SPLIT_CONFIG $DEVICE_CONFIG
python eval.py --arch "dfp-resnet34-encoder-conv" --weights "${INPUT_DIR}/dfp-resnet34-encoder-conv/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/${INPUT_DIR}/dfp-resnet34-encoder-conv" $SPLIT_CONFIG $DEVICE_CONFIG
python eval.py --arch "dfp-resnet50" --weights "${INPUT_DIR}/dfp-resnet50/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/${INPUT_DIR}/dfp-resnet50" $SPLIT_CONFIG $DEVICE_CONFIG
python eval.py --arch "deeplabv3" --weights "${INPUT_DIR}/deeplabv3/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/${INPUT_DIR}/deeplabv3" $SPLIT_CONFIG $DEVICE_CONFIG
