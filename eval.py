from collections import defaultdict

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
    # -----------------------------------------------------------------
    # 1. Create file with overall results of segmentation
    # -----------------------------------------------------------------
    global_results = defaultdict(list)
    for name, res, _ in data:
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

    for name, res, class_names in data:
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

def points_mixed_to_csv(name, points, cls_names):
    raise NotImplementedError


def points_no_class_to_csv(name, points, cls_names):
    raise NotImplementedError


def print_points_per_class(name, points, logger):
    logger.info('\n' + name)
    for t, v in points.items():
        logger.info(f'\nthreshold: {t}')
        logger.info(f'Class,Precision,Recall')
        for i in range(len(v['Per_class']['Precision'])):
            prec = v['Per_class']['Precision'][i]
            recall = v['Per_class']['Recall'][i]
            prec = round(prec*100, 2)
            recall = round(recall*100, 2)
            logger.info(f'{i},{prec},{recall}')

        prec = v['Overall']['Precision']
        recall = v['Overall']['Recall']
        prec = round(prec*100, 2)
        recall = round(recall*100, 2)
        logger.info(f'Overall,Precision,Recall')
        logger.info(f',{prec},{recall}')

def print_points_mixed(name, points, logger):
    logger.info('\n' + name)
    for t, v in points.items():
        logger.info(f'\nthreshold: {t}')
        logger.info('class')
        for row in [*range(len(v) - 1), -1]:
            logger.info(','.join(map(str, [row, *v[row]])))

        logger.info(','.join(['class', *map(str, range(len(v) - 1)), '-1']))

def print_points_no_class(name, points, logger):
    logger.info('\n' + name)
    for t, v in points.items():
        logger.info(f'\nthreshold: {t}')
        prec = v['Precision']
        recall = v['Recall']
        prec = round(prec*100, 2)
        recall = round(recall*100, 2)
        logger.info(f'Precision,Recall')
        logger.info(f'{prec},{recall}')

def evaluate(args, log_dir, logger):

    normal_set = FloorplanSVG(args.data_path, 'test.txt', format='lmdb', lmdb_folder='cubi_lmdb/', augmentations=Compose([DictToTensor()]))
    data_loader = data.DataLoader(normal_set, batch_size=1, num_workers=0)

    checkpoint = torch.load(args.weights)
    # Setup Model
    model = get_model(args.arch, 51)
    n_classes = args.n_classes
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.cuda()

    score_seg_room = runningScore(12)
    score_seg_icon = runningScore(11)
    score_pol_seg_room = runningScore(12)
    score_pol_seg_icon = runningScore(11)
    score_junctions_per_class = pointScorePerClass(list(range(13)))
    score_junctions_mixed = pointScoreMixed(13)
    score_junctions_no_class = pointScoreNoClass()

    with torch.no_grad():
        for count, val in tqdm(enumerate(data_loader), total=len(data_loader),
                               ncols=80, leave=False):
            logger.info(f'{count} - {val["folder"]}')
            things = get_evaluation_tensors(val, model, split, logger, rotate=True)

            label, segmentation, pol_segmentation, junctions, pred_junctions = things

            score_seg_room.update(label[0], segmentation[0])
            score_seg_icon.update(label[1], segmentation[1])

            score_pol_seg_room.update(label[0], pol_segmentation[0])
            score_pol_seg_icon.update(label[1], pol_segmentation[1])

            junctions_gt = {k: junctions[k] for k in junctions if k in range(-1, 13)}
            junctions_pred = {k: pred_junctions[k] for k in pred_junctions if k in range(-1, 13)}
            distance_threshold=0.01*max(val['label'].shape[2], val['label'].shape[3]) # value from r2v

            score_junctions_per_class.update(junctions_gt, junctions_pred, distance_threshold=distance_threshold)
            score_junctions_mixed.update(junctions_gt, junctions_pred, distance_threshold=distance_threshold)
            score_junctions_no_class.update(junctions_gt, junctions_pred, distance_threshold=distance_threshold)
            if count > 5:
                break
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
    )
    res_to_csv(segmentation_data, filename="segmentation", parent_dir=".")

    points_per_class_to_csv(
        points=score_junctions_per_class.get_scores(),
        class_names=score_junctions_per_class.classes,
        filename="wall_junctions_per_class",
        parent_dir=".",
    )
    
    print_points_mixed("Wall junctions mixed", score_junctions_mixed.get_scores(), logger)
    raise NotImplemented

    print_points_no_class("Wall junctions no class", score_junctions_no_class.get_scores(), logger)
    raise NotImplemented


if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Settings for evaluation')
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use [\'hg_furukawa_original, segnet etc\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of the epochs')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file .pkl')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')

    args = parser.parse_args()

    log_dir = args.log_path + '/' + time_stamp + '/'
    os.mkdir(log_dir)
    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/eval.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f'Start: {time_stamp}')
    evaluate(args, log_dir, logger)
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logger.info(f'End: {time_stamp}')
