#!/bin/bash
echo "Checking whether directory exists"
for FILE in "dfp" "dfp-last-conv" "dfp-resnet34-conv" "dfp-resnet50-conv" "deeplabv3" "dfp-resnet34-encoder-conv"
do 
    if test -f "experiment2-weighted-loss/$FILE/model_best_val_loss_var.pkl"; then
        echo
    else
        echo "$FILE does not exist"
    fi
done

echo "Running evaluation"

python eval.py --arch "dfp" --weights "experiment2-weighted-loss/dfp/model_best_val_loss_var.pkl" --output-dir "output/results/experiment2-weighted-loss/dfp"
python eval.py --arch "dfp-last-conv" --weights "experiment2-weighted-loss/dfp-last-conv/model_best_val_loss_var.pkl" --output-dir "output/results/experiment2-weighted-loss/dfp-last-conv"
python eval.py --arch "dfp-resnet34-conv" --weights "experiment2-weighted-loss/dfp-resnet34-conv/model_best_val_loss_var.pkl" --output-dir "output/results/experiment2-weighted-loss/dfp-resnet34-conv"
python eval.py --arch "dfp-resnet34-encoder-conv" --weights "experiment2-weighted-loss/dfp-resnet34-encoder-conv/model_best_val_loss_var.pkl" --output-dir "output/results/experiment2-weighted-loss/dfp-resnet34-encoder-conv"
python eval.py --arch "deeplabv3" --weights "experiment2-weighted-loss/deeplabv3/model_best_val_loss_var.pkl" --output-dir "output/results/experiment2=weighted-loss/deeplabv3"
python eval.py --arch "dfp-resnet50" --weights "experiment2-weighted-loss/dfp-resnet50-conv/model_best_val_loss_var.pkl" --output-dir "output/results/experiment2-weighted-loss/dfp-resnet50-conv"

