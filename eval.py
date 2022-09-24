from collections import defaultdict
from genericpath import exists

import numpy as np
import pandas as pd
import os
import logging
import argparse
import torch
from datetime import datetime
from torch.utils import data
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import DictToTensor, Compose
from floortrans.metrics import get_evaluation_tensors, runningScore
from floortrans.metrics_points import pointScoreNoClass, pointScorePerClass, pointScoreMixed
from tqdm import tqdm
from typing import Dict, List, Tuple

room_cls = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
icon_cls = ["Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"]


def res_to_csv(data: Tuple, filename: str, parent_dir: str="."):
    """Dump segmentation results to CSV files.

    It creates two different files at the specified ``parent_dir`` under
    the provided ``filename`` and with suffixes ``_by_class`` or
    ``_global``, depending on whether it represents the overall results
    of the segmentation or it refers to the results discriminated by class.

    Parameters
    ----------
    data: Tuple[name: str, results: Dict, class_names: List[str]]
        The data structure to dump to the CSV file. It consists of three
        components: (1) name is the descriptive name of the result being
        dumped in the file; (2) results if a 2 dimensional tuple of the
        overall metrics and the metrics discriminated by class;
        (3) class_names is an iterable consisting of the classes names
        and whose indices match the indices in the second component of
        results.
        results = (
            {"Overall Acc": float, "Mean Acc": float, "FreqW Acc": float, "Mean IoU": float},
            { "Class IoU": Dict[str, float], "Class Acc": Dict[str, float]}
        )

    filename: str
        Base name of the files being created. It is suffixed with the
        appropriate file extension.

    parent_dir: str, defaults to current directory
        The directory to write the file onto.
    """
    # -----------------------------------------------------------------
    # 1. Create file with overall results of segmentation
    # -----------------------------------------------------------------
    global_results = defaultdict(list)
    for name, res, _ in data[:-1]:
        global_results["name"].append(name)

        # first element in the ``res`` tuple contains the overall results
        global_res = res[0]
        for metric, metric_value in global_res.items():
            global_results[metric].append(metric_value)

    # Dump to file
    pd.DataFrame(global_results).to_csv(f"{parent_dir}/{filename}_global.csv", index=False)

    # -----------------------------------------------------------------
    # 2. Create file with segmentation results discriminated by class
    # -----------------------------------------------------------------
    class_results = defaultdict(list)

    for name, res, class_names in data[:-1]:
        # second element in the ``res`` tuple contains the class results
        class_res = res[1]
        class_results["name"].extend([name] * len(class_names))
        class_results["class_names"].extend(class_names)

        for class_metric, class_values in class_res.items():
            for i in range(len(class_names)):
                # Note: class_values constitute a dictionary where
                # keys are text index representation of class_names
                # and values are the corresponding metric value.
                class_val = class_values[str(i)]
                class_val = round(class_val*100, 2)
                class_results[class_metric].append(class_val)

    # Dump to file
    pd.DataFrame(class_results).to_csv(f"{parent_dir}/{filename}_by_class.csv", index=False)

    # -----------------------------------------------------------------
    # 3. Create file with just MAE for heatmap prediction
    # -----------------------------------------------------------------
    global_results = defaultdict(list)
    name, mean_iou = data[-1]
    global_results["name"].append(name)
    global_results["Mean IoU"].append(mean_iou)
    pd.DataFrame(global_results).to_csv(f"{parent_dir}/{filename}_heatmaps.csv", index=False)

def points_per_class_to_csv(
        points: Dict[int, dict],
        class_names: List[str],
        filename: str,
        parent_dir: str=".",
    ):

    results = defaultdict(list)

    for threshold, metric_values in points.items():
        # metric_values = {
        #   'Per_class': {'Recall': recall, 'Precision': precision},
        #   'Overall': {'Recall': avg_recall, 'Precision': avg_precision}
        # }

        # ------------------------------------------------------------
        # 1. Process overall metrics first
        # ------------------------------------------------------------
        results["threshold"].append(threshold)
        results["class"].append("overall")

        for metric, overall_value in metric_values["Overall"].items():
            overall_value = round(overall_value * 100, 2)
            results[metric].append(overall_value)

        # 2. Process per class
        per_class_metrics = metric_values["Per_class"]
        for i, class_name in enumerate(class_names):
            results["threshold"].append(threshold)
            results["class"].append(class_name)
            # metric will be either "Recall" or "Precision"
            # Using a for loop ensures that if we aim to extend metrics
            # with additional metrics, this script will work the same :)
            class_values = per_class_metrics[metric]

            for metric, class_values in per_class_metrics.items():
                class_value = class_values[i]
                class_value = round(class_value * 100, 2)
                results[metric].append(class_value)

        pd.DataFrame(results).to_csv(f"{parent_dir}/{filename}.csv", index=False)


def points_mixed_to_csv(
    points: Dict[int, dict],
    class_names: List[str],
    filename: str,
    parent_dir: str=".",
):
    results = defaultdict(list)

    for threshold, metric_values in points.items():
        for i, class_name in enumerate(class_names):
            # Class name
            results["class"].append(class_name)
            results["threshold"].append(threshold)

            # Confusion matrix for the specified class
            for j, cls in enumerate(class_names):
                results[cls].append(metric_values[i, j])

    pd.DataFrame(results).to_csv(f"{parent_dir}/{filename}.csv", index=False)


def points_no_class_to_csv(points, filename: str, parent_dir: str="."):
    results = defaultdict(list)

    for threshold, metric_values in points.items():
        results["threshold"].append(threshold)

        for metric_name, metric_val in metric_values.items():
            metric_val = round(metric_val * 100, 2)
            results[metric_name].append(metric_val)

    pd.DataFrame(results).to_csv(f"{parent_dir}/{filename}.csv", index=False)


def evaluate(args, log_dir, logger, output_dir: str, device="cpu"):
    if device is None:
        print(device)
        device = "cuda" if torch.cuda.is_available() else "cpu"

    normal_set = FloorplanSVG(args.data_path, args.split, format='lmdb', lmdb_folder='cubi_lmdb/',
                              augmentations=Compose([DictToTensor()]), return_heatmaps=True)
    data_loader = data.DataLoader(normal_set, batch_size=1, num_workers=0)

    checkpoint = torch.load(args.weights)
    # Setup Model
    split = [21, 12, 11]
    if args.arch != 'hg_furukawa_original':
        print("\n\n\n\n", args.arch)
        model = get_model(args.arch, args.n_classes)
    else:
        model = get_model(args.arch, 51)
        model.conv4_ = torch.nn.Conv2d(256, args.n_classes, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(args.n_classes, args.n_classes, kernel_size=4, stride=4)

    model.load_state_dict(checkpoint['model_state'])

    model.eval()
    model.to(device)

    score_seg_room = runningScore(12)
    score_seg_icon = runningScore(11)
    score_pol_seg_room = runningScore(12)
    score_pol_seg_icon = runningScore(11)
    score_junctions_per_class = pointScorePerClass(list(range(13)))
    score_junctions_mixed = pointScoreMixed(13)
    score_junctions_no_class = pointScoreNoClass()

    absolute_error = n_heatmap_pixels = 0

    with torch.no_grad():
        for count, val in tqdm(enumerate(data_loader), total=len(data_loader),
                               ncols=80, leave=False):
            logger.info(f'{count} - {val["folder"]}')
            things = get_evaluation_tensors(val, model, split, logger, rotate=True)

            label, segmentation, pol_segmentation, junctions, pred_junctions, heatmap_error = things

            score_seg_room.update(label[0], segmentation[0])
            score_seg_icon.update(label[1], segmentation[1])

            score_pol_seg_room.update(label[0], pol_segmentation[0])
            score_pol_seg_icon.update(label[1], pol_segmentation[1])

            junctions_gt = {k: junctions[k] for k in junctions if k in range(-1, 13)}
            junctions_pred = {k: pred_junctions[k] for k in pred_junctions if k in range(-1, 13)}
            distance_threshold=0.02*max(val['label'].shape[2], val['label'].shape[3]) # value from r2v

            score_junctions_per_class.update(junctions_gt, junctions_pred, distance_threshold=distance_threshold)
            score_junctions_mixed.update(junctions_gt, junctions_pred, distance_threshold=distance_threshold)
            score_junctions_no_class.update(junctions_gt, junctions_pred, distance_threshold=distance_threshold)

            absolute_error += heatmap_error
            n_heatmap_pixels += val['label'][0][:21].numel()


    csv_kwargs = {"parent_dir": output_dir}
    # Note: Segmentation data is organized as tuples of:
    # (name, res, cls_names: List[str]), where
    # - `name` is the descriptive name of the segmentation
    # - `res` is a 2-dim tuple with the following format:
    #    {"Overall Acc": v1, "Mean Acc": v2, "FreqW Acc": v3, "Mean IoU": v4},
    #    { "Class IoU": List[values], "Class Acc": List[values], }
    #
    segmentation_data = (
        ("Room segmentation", score_seg_room.get_scores(), room_cls),
        ("Room polygon segmentation", score_pol_seg_room.get_scores(), room_cls),
        ("Icon segmentation", score_seg_icon.get_scores(), icon_cls),
        ("Icon polygon segmentation", score_pol_seg_icon.get_scores(), icon_cls),
        ("Junction Prediction", absolute_error/n_heatmap_pixels),
    )
    res_to_csv(segmentation_data, filename="segmentation", **csv_kwargs)

    points_per_class_to_csv(
        points=score_junctions_per_class.get_scores(),
        class_names=score_junctions_per_class.classes,
        filename="wall_junctions_per_class",
        **csv_kwargs
    )

    points_mixed_to_csv(
        points=score_junctions_mixed.get_scores(),
        class_names=score_junctions_mixed.classes,
        filename="wall_junctions_mixed",
        **csv_kwargs
    )

    points_no_class_to_csv(
        points=score_junctions_no_class.get_scores(),
        filename="wall_junctions_no_class",
        **csv_kwargs,
    )

if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Settings for evaluation')
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use [\'hg_furukawa_original, segnet etc\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--split', nargs='?', type=str, default='test.txt',
                        help='Dataset split')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of the epochs')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file .pkl')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi',
                        help='Path to log directory')
    # We need an output dir so that the files do not overwrite each other.
    parser.add_argument('--output-dir', type=str, default='outputs/results')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_dir = args.log_path + '/' + time_stamp + '/'
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/eval.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f'Start: {time_stamp}')
    evaluate(args, log_dir, logger, args.output_dir)
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logger.info(f'End: {time_stamp}')
