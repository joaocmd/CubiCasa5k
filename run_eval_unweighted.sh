#!/bin/bash
echo "Checking whether directory exists"
for FILE in "baseline" "2022-08-09-10 51 42-dfp-experiment" "2022-08-14-09 16 10-dfp-last-conv-experiment" "2022-08-25-11 35 22-dfp-resnet34-conv" "2022-08-25-18 11 52-dfp-resnet50-conv" "2022-08-26-10 22 12-deeplabv3" "2022-08-29-23 30 37-dfp-resnet34-encoder-conv"
do 
    if test -f "experiment1-uncertainty-loss/$FILE/model_best_val_loss_var.pkl"; then
        echo
    else
        echo "$FILE does not exist"
    fi
done

echo "Running evaluation"
# python eval.py --arch hg_furukawa_original --weights "experiment1-uncertainty-loss/baseline/model_best_val_loss_var.pkl" --output-dir "output/results/experiment1-uncertainty-loss/baseline"
python eval.py --arch "dfp" --weights "experiment1-uncertainty-loss/2022-08-09-10 51 42-dfp-experiment/model_best_val_loss_var.pkl" --output-dir "output/results/experiment1-uncertainty-loss/2022-08-09-10 51 42-dfp-experiment"
python eval.py --arch "dfp-last-conv" --weights "experiment1-uncertainty-loss/2022-08-14-09 16 10-dfp-last-conv-experiment/model_best_val_loss_var.pkl" --output-dir "output/results/experiment1-uncertainty-loss/2022-08-14-09 16 10-dfp-last-conv-experiment"
python eval.py --arch "dfp-resnet34-conv" --weights "experiment1-uncertainty-loss/2022-08-25-11 35 22-dfp-resnet34-conv/model_best_val_loss_var.pkl" --output-dir "output/results/experiment1-uncertainty-loss/2022-08-25-11 35 22-dfp-resnet34-conv"
python eval.py --arch "dfp-resnet34-encoder-conv" --weights "experiment1-uncertainty-loss/2022-08-29-23 30 37-dfp-resnet34-encoder-conv/model_best_val_loss_var.pkl" --output-dir "output/results/experiment1-uncertainty-loss/2022-08-29-23 30 37-dfp-resnet34-encoder-conv"
python eval.py --arch "dfp-resnet50" --weights "experiment1-uncertainty-loss/2022-08-25-11 35 22-dfp-resnet50-conv/model_best_val_loss_var.pkl" --output-dir "output/results/experiment1-uncertainty-loss/2022-08-25-11 35 22-dfp-resnet50-conv"
python eval.py --arch "deeplabv3" --weights "experiment1-uncertainty-loss/2022-08-26-10 22 12-deeplabv3/model_best_val_loss_var.pkl" --output-dir "output/results/experiment1-uncertainty-loss/2022-08-26-10 22 12-deeplabv3"