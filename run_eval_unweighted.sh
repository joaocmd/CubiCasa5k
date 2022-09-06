#!/bin/bash
echo "Checking whether directory exists"
for FILE in "baseline" "2022-08-09-10 51 42-dfp-experiment" "2022-08-14-09 16 10-dfp-last-conv-experiment" "2022-08-25-11 35 22-dfp-resnet34-conv" "2022-08-25-11 35 22-dfp-resnet50-conv" "2022-08-26-10 22 12-deeplabv3" "2022-08-29-23 30 37-dfp-resnet34-encoder-conv"
do 
    if test -f "experiment1-uncertainty-loss/$FILE/model_best_val_loss_var.pkl"; then
        echo "experiment1-uncertainty-loss/$FILE/model_best_val_loss_var.pkl exists"
    else
        echo "$FILE does not exist"
    fi
done

echo "Running evaluation"
for FILE in "baseline" "2022-08-09-10 51 42-dfp-experiment" "2022-08-14-09 16 10-dfp-last-conv-experiment" "2022-08-25-11 35 22-dfp-resnet34-conv" "2022-08-25-11 35 22-dfp-resnet50-conv" "2022-08-26-10 22 12-deeplabv3" "2022-08-29-23 30 37-dfp-resnet34-encoder-conv"
do 
    if test -f "experiment1-uncertainty-loss/$FILE/model_best_val_loss_var.pkl"; then
        python eval.py --weights python eval.py --weights "experiment1-uncertainty-loss/$FILE/model_best_val_loss_var.pkl" --output_dir "output/results/experiment1-uncertainty-loss/$FILE"
    else
        echo "Couldn't run for $FILE"
    fi
done
