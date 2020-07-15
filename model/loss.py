#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelLoss(nn.Module):
    def __init__(self, weight_cls, weight_angle, weight_diou, weight_embed, size_average=True, use_sigmoid=False):
        super(ModelLoss, self).__init__()
        self.weight_cls = weight_cls
        self.weight_angle = weight_angle
        self.weight_diou = weight_diou
        self.weight_embed = weight_embed
        self.size_average = size_average
        self.use_sigmoid = use_sigmoid

    def forward(self, gt_score, predict_score, gt_geo, predict_geo, gt_embedding, predict_embedding, ignored_map):
        """ RBox Loss """
        # if torch.sum(gt_score) < 1:
        #     return torch.sum(predict_score + predict_geo) * 0

        loss_cls = self.get_dice_loss(gt_score, predict_score * (1 - ignored_map))
        iou_loss_map, angle_loss_map = self.get_geo_loss(gt_geo, predict_geo)

        loss_diou = torch.sum(iou_loss_map * gt_score) / (torch.sum(gt_score) + 1e-5)
        loss_ang = torch.sum(angle_loss_map * gt_score) / (torch.sum(gt_score) + 1e-5)

        """ Cosine loss: 1.0 - (y.x / |y|*|x|) """
        if self.use_sigmoid:
            loss_embed = torch.sum(1.0 - F.cosine_similarity(torch.sigmoid(predict_embedding), gt_embedding))
        else:
            loss_embed = torch.sum(1.0 - F.cosine_similarity(predict_embedding, gt_embedding))
        if self.size_average:
            loss_embed = loss_embed / predict_embedding.data.shape[1]

        loss_all = self.weight_cls * loss_cls + self.weight_angle * loss_ang + self.weight_diou * loss_diou + \
                   self.weight_embed * loss_embed
        return loss_all, loss_cls, loss_ang, loss_diou, loss_embed

    def get_dice_loss(self, gt_score, pred_score):
        inter = torch.sum(gt_score * pred_score)
        union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
        return 1. - (2 * inter / union)

    def get_geo_loss(self, gt_geo, predict_geo):
        d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
        d1_predict, d2_predict, d3_predict, d4_predict, angle_predict = torch.split(predict_geo, 1, 1)
        # calculate angle loss
        angle_loss_map = 1 - torch.cos(angle_predict - angle_gt)
        # calculate location loss
        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_predict = (d1_predict + d2_predict) * (d3_predict + d4_predict)
        w_union = torch.min(d3_gt, d3_predict) + torch.min(d4_gt, d4_predict)
        h_union = torch.min(d1_gt, d1_predict) + torch.min(d2_gt, d2_predict)
        area_intersect = w_union * h_union
        area_union = area_gt + area_predict - area_intersect
        iou = (area_intersect + 1.0) / (area_union + 1.0)

        w_center = (d4_gt - d3_gt) / 2 - (d4_predict - d3_predict) / 2
        h_center = (d2_gt - d1_gt) / 2 - (d2_predict - d1_predict) / 2
        inter_diag = (w_center ** 2) + (h_center ** 2)
        w_close = torch.max(d3_gt, d3_predict) + torch.max(d4_gt, d4_predict)
        h_close = torch.max(d1_gt, d1_predict) + torch.max(d2_gt, d2_predict)
        outer_diag = (w_close ** 2) + (h_close ** 2)
        diou = iou - (inter_diag + 1.0) / (outer_diag + 1.0)
        diou_loss_map = -torch.log((diou + 1.0) / 2.0)
        return diou_loss_map, angle_loss_map
