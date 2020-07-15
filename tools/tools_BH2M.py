#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

__author__ = 'peng zhao'

import os
import cv2
import tqdm
from xml.dom.minidom import parse

EMBEDDING_UNI_GRAMS = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9') + 1))]


def word_filter(word):
    word = str(word).lower()
    for remove_char in [w for w in word if w not in EMBEDDING_UNI_GRAMS]:
        word = word.replace(remove_char, '')
    return word


def xml2txt(mode, is_vis=False):
    pre_path = "/home/zhaopeng/WordSpottingDatasets/"
    data_name = "BH2M"
    img_path = os.path.join(pre_path, data_name, mode, "images")
    label_path = os.path.join(pre_path, data_name, mode, "labels")
    xml_path = os.path.join(pre_path, data_name, mode, "xml")
    vis_path = os.path.join(pre_path, data_name, mode, "vis")
    all_words_file = os.path.join(pre_path, data_name, mode, "all_words.txt")
    qry_words_count_file = os.path.join(pre_path, data_name, mode, "qry_words_count.txt")

    font = cv2.FONT_HERSHEY_SIMPLEX
    print("读取XML标注文件...")
    file_label_dict = {}
    for xml_name in tqdm.tqdm(sorted(os.listdir(xml_path))):
        dom_tree = parse(os.path.join(xml_path, xml_name))
        root_node = dom_tree.documentElement
        for word in root_node.getElementsByTagName("Word"):
            coords = word.getElementsByTagName("Coords")[0]
            points = str(coords.getAttribute("points")).split(' ')
            unicode = word.getElementsByTagName("Unicode")[0]
            content = word_filter(unicode.childNodes[0].data)
            if not content.strip():  # 字符串为空
                continue

            x1, y1 = [int(p) for p in points[0].split(',')]
            x2, y2 = [int(p) for p in points[1].split(',')]
            x3, y3 = [int(p) for p in points[2].split(',')]
            x4, y4 = [int(p) for p in points[3].split(',')]

            img_name = str(xml_name).split('.')[0]
            if img_name not in file_label_dict:
                file_label_dict[img_name] = []
            file_label_dict[img_name].append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                              "x3": x3, "y3": y3, "x4": x4, "y4": y4, "content": content})
    print("读取XML标注文件完成：{0}个文件".format(len(file_label_dict)))

    qry_words_count_dict = {}
    all_words_f = open(all_words_file, 'w', encoding='utf-8')
    print("生成txt文件...")
    for img_name in tqdm.tqdm(sorted(os.listdir(img_path))):
        img_name = str(img_name).split('.')[0]
        if is_vis:
            im = cv2.imread(os.path.join(img_path, "{0}.jpg".format(img_name)))
        with open(os.path.join(label_path, "{0}.txt".format(img_name)), 'w', encoding='utf-8') as f:
            for v in file_label_dict[img_name]:
                all_words_f.write("{0}\n".format(v["content"]))
                if v["content"] not in qry_words_count_dict:
                    qry_words_count_dict[v["content"]] = 0
                qry_words_count_dict[v["content"]] += 1
                f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n"
                        .format(v["x1"], v["y1"], v["x2"], v["y2"], v["x3"], v["y3"], v["x4"], v["y4"], v["content"]))
                if is_vis:
                    cv2.rectangle(im, (v["x1"], v["y1"]), (v["x3"], v["y3"]), (60, 20, 220), 2)
                    cv2.putText(im, v["content"], (v["x1"] - 10, v["y1"] - 10), font, 1, (0, 69, 255), 1)
        if is_vis:
            cv2.imwrite(os.path.join(vis_path, "{0}.jpg".format(img_name)), im)
    all_words_f.close()

    with open(qry_words_count_file, 'w', encoding='utf-8') as f:
        for key, value in qry_words_count_dict.items():
            f.write("{0}: {1}\n".format(key, value))


if __name__ == "__main__":
    # 转换训练数据集
    xml2txt(mode="train")
    # 转换测试数据集
    xml2txt(mode="test")
