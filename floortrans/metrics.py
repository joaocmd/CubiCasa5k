# Adapted from score written by wkentaro and meetshah1995
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import copy
import numpy as np
import math
import torch
from torch.nn.functional import sigmoid, softmax, interpolate
from skimage import draw
from floortrans import post_prosessing
from floortrans.loaders.augmentations import RotateNTurns
from floortrans.plotting import shp_mask

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        class_list = [str(i) for i in range(self.n_classes)]
        cls_acc = dict(zip(class_list, acc_cls))
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(class_list, iu))

        return (
            {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
            },
            {
                "Class IoU": cls_iu,
                "Class Acc": cls_acc
            }
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def up_sample_predictions(pred, size):
    width = size[1]
    height = size[0]
    pred_count, channels, pred_height, pred_width = pred.shape

    if width > height:
        ratio_of_pad = (1 - float(height) / float(width)) / 2.0
        pad = int(math.floor(pred_height * ratio_of_pad))
        pred = pred[:, :, pad:-pad, :]
    else:
        ratio_of_pad = (1 - float(width) / float(height)) / 2.0
        pad = int(math.floor(pred_width * ratio_of_pad))
        pred = pred[:, :, :, pad:-pad]

    # In the paper they use bi-cubic interpolation here also...
    # Don't understand why
    pred = transform.resize(pred, (pred_count, channels, height, width),
                            order=3, mode='constant', anti_aliasing=False)

    return pred


def get_px_acc(pred, target, input_slice, sub=1):
    pred_arr = torch.split(pred, input_slice)
    heatmap_pred, rooms_pred, icons_pred = pred_arr
    rooms_pred = softmax(rooms_pred, 0).argmax(0)
    rooms_target = target[input_slice[0]].type(torch.cuda.LongTensor) - sub
    rooms_pos = torch.eq(rooms_pred, rooms_target).sum()

    icons_target = target[input_slice[0]+1].type(torch.cuda.LongTensor) - sub
    icons_pred = softmax(icons_pred, 0).argmax(0)
    icons_pos = torch.eq(icons_pred, icons_target).sum()

    return rooms_pos, icons_pos


def pixel_accuracy(label, pred):
    total_px = label.shape[0] * label.shape[1]
    sum_correct = np.equal(label, pred).sum()

    return float(sum_correct) / float(total_px)


def polygons_to_tensor(polygons_val, types_val, room_polygons_val, room_types_val, size, split=[12, 11]):
    ten = np.zeros((sum(split), size[0], size[1]))

    for i, pol_type in enumerate(room_types_val):
        mask = shp_mask(room_polygons_val[i], np.arange(size[1]), np.arange(size[0]))
        ten[pol_type['class']][mask] = 1

    for i, pol_type in enumerate(types_val):
        if pol_type['type'] == 'icon':
            d = split[0]
        else:
            d = 0
        jj, ii = draw.polygon(polygons_val[i][:, 1], polygons_val[i][:, 0])
        # ten[pol_type['class'] + d][jj, ii] = 1
        ten[pol_type['class'] + d][jj[0]:jj[-1]+1, ii[0]:ii[-1]+1] = 1

    return ten


def extract_points(heatmaps, threshold = 0.09):
    points = {}
    for i in range(len(heatmaps)):
        # info = [int(i / 4), int(i % 4)]
        info = i
        p = extract_local_max(heatmaps[i], info, threshold, close_point_suppression=True)
        points[i] = p

    points[-1] = extract_local_max(np.max(heatmaps[:13], axis=0), -1, 0.2, close_point_suppression=True)
    return points

def extract_local_max(mask_img, info, heatmap_value_threshold=0.2,
                      close_point_suppression=False, gap=10):
    mask = copy.deepcopy(mask_img)
    height, width = mask.shape
    points = []

    for i in range(1000): # HACK: should be while True
        if i == 999:
            print('extract_local_max: SHOULD NOT HAVE HAPPENED')

        index = np.argmax(mask)
        y, x = np.unravel_index(index, mask.shape)
        max_value = mask[y, x]
        if max_value < heatmap_value_threshold:
            return points

        points.append([int(x), int(y)] + [info, max_value])

        maximum_suppression_iterative(mask, x, y, heatmap_value_threshold)
        if close_point_suppression:
            mask[max(y - gap, 0):min(y + gap, height - 1),
                 max(x - gap, 0):min(x + gap, width - 1)] = 0

    return points

def maximum_suppression(mask, x, y, heatmap_value_threshold):
    height, width = mask.shape
    value = mask[y][x]
    mask[y][x] = -1
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta in deltas:
        neighbor_x = x + delta[0]
        neighbor_y = y + delta[1]
        if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= width or neighbor_y >= height:
            continue
        neighbor_value = mask[neighbor_y][neighbor_x]
        if neighbor_value <= value and neighbor_value > heatmap_value_threshold:
            maximum_suppression(mask, neighbor_x, neighbor_y,
                               heatmap_value_threshold)
            pass
        continue

def maximum_suppression_iterative(mask, x, y, heatmap_value_threshold):
    height, width = mask.shape

    stack = [(x, y)]
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(stack) != 0:
        x, y = stack.pop(0)
        value = mask[y][x]
        if value == -1:
            continue
        mask[y][x] = -1

        for delta in deltas:
            neighbor_x = x + delta[0]
            neighbor_y = y + delta[1]
            if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= width or neighbor_y >= height:
                continue

            neighbor_value = mask[neighbor_y][neighbor_x]

            if neighbor_value <= value and neighbor_value > heatmap_value_threshold:
                stack.append((neighbor_x, neighbor_y))

def get_evaluation_tensors(val, model, split, logger, rotate=True, n_classes=44):
    images_val = val['image'].cuda()
    labels_val = val['label']
    junctions_val = {i: [[int(x), int(y)] for x, y in v] for i, v in val['heatmaps'].items()}

    height = labels_val.shape[2]
    width = labels_val.shape[3]
    img_size = (height, width)

    if rotate:
        rot = RotateNTurns()
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(images_val, 'tensor', forward)
            pred = model(rot_image)
            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]
            # logger.info("flip: " + str(i))

        prediction = torch.mean(prediction, 0, True)
    else:
        prediction = model(images_val)

    heatmaps, rooms, icons = post_prosessing.split_prediction(
        prediction, img_size, split)

    predicted_points = extract_points(heatmaps)
    rooms_seg = np.argmax(rooms, axis=0)
    icons_seg = np.argmax(icons, axis=0)

    all_opening_types = [1, 2]  # Window, Door
    polygons, types, room_polygons, room_types, _ = post_prosessing.get_polygons(
        # ------------------------------------------------------------------
        # Update 2022/09/09: This method keeps breaking w/ max depth recursion
        # we'll update both the threshold (forcing it to prune branches)
        # sooner, and also use an iterative version of the recursive
        # rationale, with a max depth call of 999, after which we assume
        # there's no pixel that satisfies this threshold.
        # ------------------------------------------------------------------
        # (heatmaps, rooms, icons), 0.4, all_opening_types)
        (heatmaps, rooms, icons), 0.9, all_opening_types)

    logger.info("Prediction post processing done")

    predicted_classes = polygons_to_tensor(
        polygons, types, room_polygons, room_types, img_size)
    
    pol_rooms = np.argmax(predicted_classes[:split[1]], axis=0)
    pol_icons = np.argmax(predicted_classes[split[1]:], axis=0)
    
    return (labels_val[0, 21:].data.numpy(),
            np.concatenate(([rooms_seg], [icons_seg]), axis=0),
            np.concatenate(([pol_rooms], [pol_icons]), axis=0),
            junctions_val,
            predicted_points)
