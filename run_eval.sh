#!/bin/bash

echo "Running evaluation"
INPUT_DIR=experiment1-uncertainty-loss
OUTPUT_DIR=output/results/${INPUT_DIR}
MODEL_TYPE=model_best_val_loss_var.pkl

echo "Eval Furukawa Original baseline"
python eval.py --arch hg_furukawa_original --weights "${INPUT_DIR}/baseline/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/baseline"
    
for INPUT_DIR in "experiment1-uncertainty-loss" "experiment2-weighted-loss"
do
    python eval.py --arch "dfp" --weights "${INPUT_DIR}/dfp/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/dfp"
    python eval.py --arch "dfp-last-conv" --weights "${INPUT_DIR}/dfp-last-conv/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/dfp-last-conv"
    python eval.py --arch "dfp-resnet34" --weights "${INPUT_DIR}/dfp-resnet34/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/dfp-resnet34"
    python eval.py --arch "dfp-resnet34-encoder-conv" --weights "${INPUT_DIR}/dfp-resnet34-encoder-conv/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/dfp-resnet34-encoder-conv"
    python eval.py --arch "dfp-resnet50" --weights "${INPUT_DIR}/dfp-resnet50/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/dfp-resnet50"
    python eval.py --arch "deeplabv3" --weights "${INPUT_DIR}/deeplabv3/${MODEL_TYPE}" --output-dir "${OUTPUT_DIR}/deeplabv3"
done