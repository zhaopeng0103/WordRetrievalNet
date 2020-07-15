#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from model.fpn import FPN, BBoxOutputLayer, EmbeddingOutputLayer
from model.loss import ModelLoss


# WordRetrievalModel
class WordRetrievalModel(nn.Module):
    def __init__(self, n_out, backbone="resnet50", pre_trained=True):
        super(WordRetrievalModel, self).__init__()
        self.fpn = FPN(backbone=backbone, out_channels=256, pre_trained=pre_trained)
        self.bbox_output = BBoxOutputLayer()
        self.embedding_output = EmbeddingOutputLayer(n_out=n_out)

    def forward(self, input_x):
        fpn_out = self.fpn(input_x)  # torch.Size([1, 1024, 128, 128])
        return self.bbox_output(fpn_out), self.embedding_output(fpn_out)
