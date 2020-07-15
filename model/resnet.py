#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# Resnet Graph
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes * BasicBlock.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, input_x):
        residual = input_x
        out = self.relu(self.bn1(self.conv1(input_x)))
        out = self.bn2(self.conv2(out)) + (self.down_sample(input_x) if self.down_sample is not None else residual)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, input_x):
        residual = input_x
        out = self.relu(self.bn1(self.conv1(input_x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + (self.down_sample(input_x) if self.down_sample is not None else residual)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.block = block
        self.layers = layers

        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.C2 = self.make_layer(block=self.block, planes=64, blocks=self.layers[0])
        self.C3 = self.make_layer(block=self.block, planes=128, blocks=self.layers[1], stride=2)
        self.C4 = self.make_layer(block=self.block, planes=256, blocks=self.layers[2], stride=2)
        self.C5 = self.make_layer(block=self.block, planes=512, blocks=self.layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, down_sample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_pre_trained_model(self, model_url):
        pre_trained_model = model_zoo.load_url(model_url)
        state = self.state_dict()
        for key in state.keys():
            if key in pre_trained_model.keys():
                state[key] = pre_trained_model[key]
        self.load_state_dict(state)
        print('======> Loading pre_trained model from: {0}'.format(model_url))

    def forward(self, input_x):
        out = input_x
        c1_out = self.C1(out)  # torch.Size([1, 64, 224, 224])
        c2_out = self.C2(c1_out)  # torch.Size([1, 256, 224, 224])
        c3_out = self.C3(c2_out)  # torch.Size([1, 512, 112, 112])
        c4_out = self.C4(c3_out)  # torch.Size([1, 1024, 56, 56])
        c5_out = self.C5(c4_out)  # torch.Size([1, 2048, 28, 28])
        return c1_out, c2_out, c3_out, c4_out, c5_out


def resnet18(pre_trained=True):
    """ Constructs a ResNet-18 model. """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pre_trained:
        model.load_pre_trained_model(model_urls['resnet18'])
    return model


def resnet34(pre_trained=True):
    """ Constructs a ResNet-34 model. """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pre_trained:
        model.load_pre_trained_model(model_urls['resnet34'])
    return model


def resnet50(pre_trained=True):
    """ Constructs a ResNet-50 model. """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pre_trained:
        model.load_pre_trained_model(model_urls['resnet50'])
    return model


def resnet101(pre_trained=True):
    """ Constructs a ResNet-101 model. """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pre_trained:
        model.load_pre_trained_model(model_urls['resnet101'])
    return model


def resnet152(pre_trained=True):
    """ Constructs a ResNet-152 model. """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pre_trained:
        model.load_pre_trained_model(model_urls['resnet152'])
    return model
