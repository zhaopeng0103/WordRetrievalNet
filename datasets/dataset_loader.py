#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils import data

from datasets.bbox_func import *
from datasets.embedding_func import *
from config import data_cfg


def extract_ver_lab_anno(gt_file):
    """ extract vertices info from txt lines
        Input:
            gt_file        : gt file path
        Output:
            vertices       : vertices of text regions <numpy.ndarray, (n,8)>
            labels         : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
            annotations    : annotations of text regions <numpy.ndarray, (n,)>
    """
    vertices = []
    labels = []
    annotations = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().rstrip('\n').lstrip('\ufeff').strip().split(',', maxsplit=8)
        vertices.append(list([int(ver) for ver in line[:8]]))
        annotations.append(str(line[-1]).strip())
        if str(line[-1]).strip() == "" or str(line[-1]).strip() is None or len(str(line[-1]).strip()) < 1:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(vertices), np.array(labels), np.array(annotations)


def get_all_words(annotation_files):
    """ get all words from all annotation files
        Input:
            annotation_files: all gt files <list>
        Output:
            all_words       : all words list <numpy.ndarray, (n,)>
            unique_words    : unique words list <numpy.ndarray, (n,)>
    """
    all_words = np.array([])
    for annotation_file in annotation_files:
        annotations = []
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().rstrip('\n').lstrip('\ufeff').strip().split(',', maxsplit=8)
            if str(line[-1]).strip() != "" and str(line[-1]).strip() is not None:
                annotations.append(str(line[-1]).strip())
        annotations = np.array(annotations)
        all_words = np.concatenate((all_words, annotations), axis=0)
    unique_words = list(set(all_words))
    print("Get_All_Words::All words num / Unique words num is {0}/{1}".format(len(all_words), len(unique_words)))
    return all_words, unique_words


def get_map(img, vertices, labels, annotations, embeddings, scale, length, embedding_size):
    """ generate score gt and geometry gt
        Input:
            img            : PIL Image
            vertices       : vertices of text regions <numpy.ndarray, (n,8)>
            labels         : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
            annotations    : word string contents in the img
            embeddings     : all uni_grams embedding
            scale          : feature map / image
            length         : image length
            embedding_size : embedding size of given words
        Output:
            score gt, geo gt, ignored, embedding gt
    """

    score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    embedding_map = np.zeros((int(img.height * scale), int(img.width * scale), embedding_size), np.float32)

    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        if labels[i] == 0:
            ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
            continue
        if np.any(np.around(scale * vertice.reshape((4, 2))).astype(np.int32) <= 0):
            continue
        if np.any(np.around(scale * vertice.reshape((4, 2))).astype(np.int32) >= int(scale * img.height)):
            continue

        poly = np.around(scale * shrink_poly(vertice, coef=0.2).reshape((4, 2))).astype(np.int32)  # scaled & shrink
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask

        min_x = int(min(poly[0][0], poly[1][0], poly[2][0], poly[3][0]))
        max_x = int(max(poly[0][0], poly[1][0], poly[2][0], poly[3][0]))
        min_y = int(min(poly[0][1], poly[1][1], poly[2][1], poly[3][1]))
        max_y = int(max(poly[0][1], poly[1][1], poly[2][1], poly[3][1]))
        embedding_map[min_y:max_y, min_x:max_x] = embeddings[annotations[i]]

    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)

    return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo_map).permute(2, 0, 1), \
           torch.Tensor(ignored_map).permute(2, 0, 1), torch.Tensor(embedding_map).permute(2, 0, 1)


class CustomDataSetRBox(data.Dataset):
    def __init__(self, data_select, scale=0.25, max_img_length=512, long_size=2048):
        super(CustomDataSetRBox, self).__init__()
        self.scale = scale
        self.max_img_length = max_img_length
        self.long_size = long_size

        # init train gt & img information
        self.train_img_path, self.train_gt_path = data_select["train_img_path"], data_select["train_gt_path"]
        self.train_img_files = [os.path.join(self.train_img_path, img_file) for img_file in sorted(os.listdir(self.train_img_path))]
        self.train_gt_files = [os.path.join(self.train_gt_path, gt_file) for gt_file in sorted(os.listdir(self.train_gt_path))]
        self.train_words, self.train_unique_words = get_all_words(self.train_gt_files)

        # init test gt & img information
        self.test_img_path, self.test_gt_path = data_select["test_img_path"], data_select["test_gt_path"]
        self.test_img_files = [os.path.join(self.test_img_path, img_file) for img_file in sorted(os.listdir(self.test_img_path))]
        self.test_gt_files = [os.path.join(self.test_gt_path, gt_file) for gt_file in sorted(os.listdir(self.test_gt_path))]
        self.test_words, _ = get_all_words(self.test_gt_files)

        self.unique_words = list(set(np.concatenate((self.train_words, self.test_words), axis=0)))
        self.embedding_size, self.words_embeddings = build_embedding_descriptor(self.unique_words)

    def __len__(self):
        return len(self.train_img_files)

    def __getitem__(self, index):
        vertices, labels, annotations = extract_ver_lab_anno(self.train_gt_files[index])
        img = Image.open(self.train_img_files[index])
        try:
            img = img.convert("RGB")
        except:
            print("===> convert fail: ", self.train_img_files[index])

        img, vertices = random_scale(img, vertices, long_size=self.long_size)
        img, vertices = rotate_img(img, vertices)
        img, vertices = crop_img(img, vertices, labels, self.max_img_length)
        transform = transforms.Compose([transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        score_map, geo_map, ignored_map, embedding_map = get_map(img, vertices, labels, annotations, self.words_embeddings,
                                                                 self.scale, self.max_img_length, self.embedding_size)
        return transform(img), score_map, geo_map, ignored_map, embedding_map


if __name__ == "__main__":
    train_set = CustomDataSetRBox(data_cfg[0], max_img_length=512)
    train_set.__getitem__(0)
