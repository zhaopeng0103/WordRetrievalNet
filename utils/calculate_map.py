#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


import tqdm
import numpy as np
import scipy.sparse as spa
from multiprocessing import Pool
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist
from post_processing import lanms
# from post_processing import locality_aware_nms as nms_locality


def average_precision_segfree(res, t, o, sinds, n_relevant, ot):
    """
        Computes the average precision
        res: sorted list of labels of the proposals, aka the results of a query.
        t: transcription of the query
        o: overlap matrix between the proposals and gt_boxes.
        sinds: The gt_box with which the proposals overlaps the most.
        n_relevant: The number of relevant retrievals in ground truth dataset
        ot: overlap_threshold
    """
    correct_label = res == t

    # The highest overlap between a proposal and a ground truth box
    tmp = []
    covered = []
    for i in range(len(res)):
        if sinds[i] not in covered:  # if a ground truth box has been covered, mark proposal as irrelevant
            tmp.append(o[i, sinds[i]])
            if o[i, sinds[i]] >= ot and correct_label[i]:
                covered.append(sinds[i])
        else:
            tmp.append(0.0)

    covered = np.array(covered)
    tmp = np.array(tmp)
    relevance = correct_label * (tmp >= ot)
    rel_cumsum = np.cumsum(relevance, dtype=float)
    precision = rel_cumsum / np.arange(1, relevance.size + 1)

    if n_relevant > 0:
        ap = (precision * relevance).sum() / n_relevant
    else:
        ap = 0.0
    return ap, covered


def hh(arg):
    query, t, db, db_targets, joint_boxes, query_nms_overlap, all_overlaps, inds, gt_targets, ot, qw = arg
    count = np.sum(db_targets == t)
    if count == 0:  # i.e., we have missed this word completely
        return 0.0, 0.0

    dists = np.squeeze(cdist(query[np.newaxis, :], db, metric="cosine"))
    sim = (dists.max()) - dists

    dets = np.hstack((joint_boxes, sim[:, np.newaxis]))
    nms_dets, pick = lanms.merge_quadrangle_n9(dets.astype('float32'), query_nms_overlap)

    I = np.argsort(dists[pick])
    res = db_targets[pick][I]  # Sort results after distance to query image
    o = all_overlaps[pick][I, :]
    sinds = inds[pick][I]
    n_relevant = np.sum(gt_targets == t)
    ap, covered = average_precision_segfree(res, t, o, sinds, n_relevant, ot)
    r = float(np.unique(covered).shape[0]) / n_relevant
    # print("===> query_word: {0}; ap: {1}; recall: {2}; n_relevant: {3}".format(qw, ap, r, n_relevant))
    return ap, r


def cal_map(queries, qtargets, db, db_targets, gt_targets, joint_boxes, all_overlaps, query_nms_overlap, ot, qbs_words, num_workers):
    inds = all_overlaps.argmax(axis=1)
    all_overlaps = spa.csr_matrix(all_overlaps)
    args = [(q, t, db, db_targets, joint_boxes, query_nms_overlap, all_overlaps, inds, gt_targets, ot, qw)
            for q, t, qw in zip(queries, qtargets, qbs_words)]
    if num_workers == 0:  # 单线程
        res = []
        for arg in tqdm.tqdm(args):
            res.append(hh(arg))
    else:  # 多线程
        p = Pool(num_workers)
        res = p.map(hh, tqdm.tqdm(args))
    return np.mean(np.array(res), axis=0)


def cal_overlap(ab, tb):
    overlaps = np.array([[overlap(b1, b2) for b2 in tb] for b1 in ab])
    return overlaps


def overlap(b1, b2):
    g = Polygon(b1[:8].reshape((4, 2)))
    p = Polygon(b2[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0.0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0.0
    else:
        return float(inter) / float(union)
