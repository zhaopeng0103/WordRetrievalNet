#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


data_cfg = [
    {
        "name": "Konzilsprotokolle",
        "train_img_path": "/home/zhaopeng/WordSpottingDatasets/Konzilsprotokolle/gen/images/",
        "train_gt_path": "/home/zhaopeng/WordSpottingDatasets/Konzilsprotokolle/gen/labels/",
        "test_img_path": "/home/zhaopeng/WordSpottingDatasets/Konzilsprotokolle/test/images/",
        "test_gt_path": "/home/zhaopeng/WordSpottingDatasets/Konzilsprotokolle/test/labels/",
    },
    {
        "name": "BH2M",
        "train_img_path": "/home/zhaopeng/WordSpottingDatasets/BH2M/gen/images/",
        "train_gt_path": "/home/zhaopeng/WordSpottingDatasets/BH2M/gen/labels/",
        "test_img_path": "/home/zhaopeng/WordSpottingDatasets/BH2M/test/images/",
        "test_gt_path": "/home/zhaopeng/WordSpottingDatasets/BH2M/test/labels/",
    },
]


global_cfg = {
    "data_cfg": data_cfg[1],
    "arch": {
        "backbone": "resnet50",
        "pre_trained": True,
    },
    "loss": {
        "weight_cls": 1.0,
        "weight_angle": 10.0,
        "weight_diou": 1.0,
        "weight_embed": 1.0,
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 1e-3,
            "weight_decay": 0.00005,
            "amsgrad": True,
        },
    },
    "lr_scheduler": {
        "type": "MultiStepLR",
        "args": {
            "milestones": [200],
            "gamma": 0.1,
        }
    },
    "trainer": {
        "seed": 2,
        "gpus": [1],
        "img_channel": 3,
        "input_size": 512,
        "long_size": 2048,
        "batch_size": 4,
        "num_workers": 16,
        "epochs": 120,
        "lr_step": [80, 100],
        "save_interval": 10,
        "display_interval": 10,
        "show_images_interval": 10,
        "resume_checkpoint": "",
        "finetune_checkpoint": "",
        "output_dir": "output",
        "tensorboard": True,
        "metrics": "map",
    },
    "tester": {
        "img_channel": 3,
        "long_size": 2048,
        "output_dir": "output",
        "cls_score_thresh": 0.9,
        "bbox_nms_overlap": 0.4,
        "query_nms_overlap": 0.9,
        "overlap_thresh": [0.25, 0.5],
        "distance_metric": "cosine",
    },
}
