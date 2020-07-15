#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from model.resnet import *

d = ['resnet18', 'resnet34','resnet50', 'resnet101', 'resnet152']


# FPN Graph
class FPN(nn.Module):
    def __init__(self, backbone, out_channels, pre_trained):
        super(FPN, self).__init__()
        assert backbone in d, 'backbone must in: {}'.format(d)
        self.backbone = globals()[backbone](pre_trained=pre_trained)
        self.out_channels = out_channels

        # Top layer: Reduce channels
        self.top_layer = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )

        # Smooth layers
        self.smooth1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )

        # Lateral layers
        self.lat_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.lat_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.lat_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=True)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=True)
        return torch.cat((p2, p3, p4, p5), 1)

    def forward(self, input_x):
        _, c2_out, c3_out, c4_out, c5_out = self.backbone(input_x)

        p5_out = self.top_layer(c5_out)

        p4_out = self._upsample_add(p5_out, self.lat_layer1(c4_out))
        p4_out = self.smooth1(p4_out)

        p3_out = self._upsample_add(p4_out, self.lat_layer2(c3_out))
        p3_out = self.smooth2(p3_out)

        p2_out = self._upsample_add(p3_out, self.lat_layer3(c2_out))
        p2_out = self.smooth3(p2_out)

        out = self._upsample_cat(p2_out, p3_out, p4_out, p5_out)
        return out  # 1024


# location & angle & score output branch
class BBoxOutputLayer(nn.Module):
    def __init__(self):
        super(BBoxOutputLayer, self).__init__()
        self.scope = 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
        )

        self.conv2_1_out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv2_2_out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv2_3_out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_x):
        out = input_x
        out = self.conv1(out)

        scoring = self.conv2_1_out(out)
        location = self.conv2_2_out(out) * self.scope
        angle = (self.conv2_3_out(out) - 0.5) * math.pi
        geometry = torch.cat((location, angle), 1)
        return scoring, geometry


# Embedding output branch
class EmbeddingOutputLayer(nn.Module):
    def __init__(self, n_out):
        super(EmbeddingOutputLayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
        )

        self.conv2_out = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=n_out, kernel_size=1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_x):
        return self.conv2_out(self.conv1(input_x))
