import numpy as np
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

room_cls = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
icon_cls = ["Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"]


def print_res(name, res, cls_names, logger):
    basic_res = res[0]
    class_res = res[1]

    basic_names = ''
    basic_values = name
    basic_res_list = ["Overall Acc", "Mean Acc", "Mean IoU", "FreqW Acc"]
    for key in basic_res_list:
        basic_names += ',' + key
        val = round(basic_res[key] * 100, 2)
        basic_values += ',' + str(val)

    logger.info(basic_names)
    logger.info(basic_values)

    basic_res_list = ["IoU", "Acc"]
    logger.info("IoU & Acc")
    for i, name in enumerate(cls_names):
        iou = class_res['Class IoU'][str(i)]
        acc = class_res['Class Acc'][str(i)]
        iou = round(iou * 100, 2)
        acc = round(acc * 100, 2)
        logger.info(name + "," + str(iou) + "," + str(acc))

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

    print_res("Room segmentation", score_seg_room.get_scores(), room_cls, logger)
    print_res("Room polygon segmentation", score_pol_seg_room.get_scores(), room_cls, logger)
    print_res("Icon segmentation", score_seg_icon.get_scores(), icon_cls, logger)
    print_res("Icon polygon segmentation", score_pol_seg_icon.get_scores(), icon_cls, logger)
    print_points_per_class("Wall junctions per class", score_junctions_per_class.get_scores(), logger)
    print_points_mixed("Wall junctions mixed", score_junctions_mixed.get_scores(), logger)
    print_points_no_class("Wall junctions no class", score_junctions_no_class.get_scores(), logger)


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
