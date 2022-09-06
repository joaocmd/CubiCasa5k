# Adapted from score written by wkentaro and meetshah1995
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
import itertools

def sqrdistance(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

def point(p):
    return (p[0], p[1])

def filter_points(pointsDict, threshold):
    points = dict()
    for k in pointsDict:
        points[k] = [v for v in pointsDict[k] if v[3] >= threshold]
    return points

class pointScoreNoClass:
    def __init__(self):
        self.scores = {t: np.array([0, 0, 0], dtype=int) for t in range(0, 101, 10)} # tp fp gt

    def update(self, gt, predicted, distance_threshold):
        for t in self.scores:
            self.scores[t] = self.update_one(gt, filter_points(predicted, t/100), distance_threshold, self.scores[t])

    def update_one(self, gt, predicted, distance_threshold, values):
        all_predicted = predicted[-1]
        all_gt = []
        for k in gt:
            all_gt += [[val[0], val[1], k] for val in gt[k]]

        pairs = itertools.product(all_predicted, all_gt)
        pairs = ([pair, sqrdistance(pair[0], pair[1])] for pair in pairs)
        pairs = (x for x in pairs if x[1] < distance_threshold)
        pairs = sorted(pairs, key=lambda p: p[1])
        pairs = [[point(p[0][0]), point(p[0][1])] for p in pairs]
        
        values[2] += len(all_gt)
        points = set(point(p) for p in all_predicted)
        while len(pairs) != 0 and len(points) != 0:
            closest = pairs[0] # it is sorted already

            values[0] += 1
            points.remove(closest[0]) # first is the predicted (from cartesian product)
            pairs = [p for p in pairs if p[0] != closest[0] and p[1] != closest[1]]

        values[1] += len(points)

        return values


    def get_scores(self):
        scores = dict()
        for t, val in self.scores.items():
            # tp fp gt
            scores[t] = {'Recall': val[0]/val[2], 'Precision': val[0]/(val[0] + val[1])}

        return scores

class pointScorePerClass:
    def __init__(self, keys):
        self.classes = keys
        self.n_classes = len(keys)
        self.scores = {t: np.zeros((len(keys), 3), dtype=int) for t in range(0, 101, 10)} # tp fp gt

    def update(self, gt, predicted, distance_threshold):
        for t in self.scores:
            self.scores[t] = self.update_one(gt, filter_points(predicted, t/100), distance_threshold, self.scores[t])

    def update_one(self, gt, predicted, distance_threshold, values):
        for k in range(self.n_classes):
            pts_gt = [tuple(pt) for pt in gt[k]]
            pts_pred = predicted[k][:]

            values[k][1] += len(pts_pred)
            values[k][2] += len(pts_gt)

            if len(pts_gt) == 0:
                continue

            for pt in pts_pred:
                p, distance = min([[p, sqrdistance(p, pt)] for p in pts_gt], key=lambda p: p[1])

                if distance < distance_threshold:
                    values[k][0] += 1
                    values[k][1] -= 1
                    pts_gt.remove(p)
                    if len(pts_gt) == 0:
                        break
        return values

    def get_scores(self):
        scores = dict()
        for t, val in self.scores.items():
            # tp fp gt
            scores[t] = self.get_score(val)

        return scores

    def get_score(self, scores):
        # tp fp gt
        # acc = self.scores[:,0] / (self.scores[:,1] + self.scores[:,2])
        lp = (scores[:,2])
        recall = np.where(lp == 0, np.nan, scores[:,0] / lp)

        # predicted positives
        pp = (scores[:,0] + scores[:,1])
        precision = np.where(pp == 0, np.nan, scores[:,0] / pp) 

        # acc = np.sum(self.scores[:,0]) / (np.sum(self.scores[:,1]) + np.sum(self.scores[:,2]))
        # acc = np.sum(self.scores[:,0]) / (np.sum(self.scores[:,2]))
        avg_recall = np.mean(recall[~np.isnan(recall)])
        avg_precision = np.mean(precision[~np.isnan(precision)])

        return {'Per_class': {'Recall': recall, 'Precision': precision},
                'Overall': {'Recall': avg_recall, 'Precision': avg_precision}}

class pointScoreMixed:
    def __init__(self, n_classes):
        self.classes = list(range(n_classes)) + [-1] 
        self.n_classes = n_classes
        self.scores = {t: np.zeros((n_classes+1, n_classes+1), dtype=int) for t in range(0, 101, 10)} # tp fp gt

    def update(self, gt, predicted, distance_threshold):
        for t in self.scores:
            self.scores[t] = self.update_one(gt, filter_points(predicted, t/100), distance_threshold, self.scores[t])

    def update_one(self, gt, predicted, distance_threshold, values):
        all_predicted = []
        for k in gt:
            all_predicted += [(val[0], val[1], k) for val in predicted[k]]

        all_gt = []
        for k in gt:
            all_gt += [(val[0], val[1], k) for val in gt[k]]

        pairs = itertools.product(all_predicted, all_gt)
        pairs = ([pair, sqrdistance(pair[0], pair[1])] for pair in pairs)
        pairs = (x for x in pairs if x[1] < distance_threshold)
        pairs = sorted(pairs, key=lambda p: p[1])
        
        remaining_pred = set(all_predicted)
        remaining_gt = set(all_gt)
        while len(pairs) != 0:
            pred, real = pairs[0][0][0], pairs[0][0][1] # it is sorted already
            values[real[2]][pred[2]] += 1
            remaining_pred.remove(pred)
            remaining_gt.remove(real)
            pairs = [p for p in pairs if p[0][0] != pred and p[0][1] != real]

        for p in remaining_pred:
            values[-1][p[2]] += 1
        for p in remaining_gt:
            values[p[2]][-1] += 1

        return values

    def get_scores(self):
        scores = dict()
        for t, val in self.scores.items():
            # tp fp gt
            scores[t] = self.get_score(val)

        return scores

    def get_score(self, scores):
        # tp fp gt
        # acc = self.scores[:,0] / (self.scores[:,1] + self.scores[:,2])
        return scores
        # recall = scores[:,0] / (scores[:,2])
        # precision = scores[:,0] / (scores[:,0] + scores[:,1])

        # # acc = np.sum(self.scores[:,0]) / (np.sum(self.scores[:,1]) + np.sum(self.scores[:,2]))
        # # acc = np.sum(self.scores[:,0]) / (np.sum(self.scores[:,2]))
        # avg_recall = np.mean(recall[~np.isnan(recall)])
        # avg_precision = np.mean(precision[~np.isnan(precision)])

        # return {'Per_class': {'Recall': recall, 'Precision': precision},
        #         'Overall': {'Recall': avg_recall, 'Precision': avg_precision}}