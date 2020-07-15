#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

__author__ = 'peng zhao'

import os
import cv2
import tqdm
from xml.dom.minidom import parse

EMBEDDING_UNI_GRAMS = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9') + 1))]
xml_file = {
    "Konzilsprotokolle_train": ["Konzilsprotokolle_Train_I_WL.xml", "Konzilsprotokolle_Train_II_WL.xml", "Konzilsprotokolle_Train_III_WL.xml"],
    "Konzilsprotokolle_test": ["Konzilsprotokolle_Test_GT_SegFree_QbS.xml"],
}


def word_filter(word):
    word = str(word).lower()
    for remove_char in [w for w in word if w not in EMBEDDING_UNI_GRAMS]:
        word = word.replace(remove_char, '')
    return word


def xml2txt(mode, is_vis=False):
    pre_path = "/home/zhaopeng/WordSpottingDatasets/"
    data_name = "Konzilsprotokolle"
    img_path = os.path.join(pre_path, data_name, mode, "images")
    label_path = os.path.join(pre_path, data_name, mode, "labels")
    xml_path = os.path.join(pre_path, data_name, mode, "xml")
    vis_path = os.path.join(pre_path, data_name, mode, "vis")
    qry_file = os.path.join(pre_path, data_name, "test", "Konzilsprotokolle_Test_QryStrings.lst")
    all_words_file = os.path.join(pre_path, data_name, mode, "all_words.txt")
    qry_words_count_file = os.path.join(pre_path, data_name, mode, "qry_words_count.txt")

    font = cv2.FONT_HERSHEY_SIMPLEX
    print("读取查询单词...")
    with open(qry_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    qry_words = [word_filter(line.strip()) for line in lines]
    print("读取查询单词完成：{0}个单词".format(len(qry_words)))

    print("读取XML标注文件...")
    file_label_dict = {}
    for xml_name in xml_file["{0}_{1}".format(data_name, mode)]:
        dom_tree = parse(os.path.join(xml_path, xml_name))
        root_node = dom_tree.documentElement
        for spot in root_node.getElementsByTagName("spot"):
            img_name = str(spot.getAttribute("image")).split('.')[0]
            word = word_filter(spot.getAttribute("word"))
            x, y = int(spot.getAttribute("x")), int(spot.getAttribute("y"))
            w, h = int(spot.getAttribute("w")), int(spot.getAttribute("h"))
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
            x1, y1, x2, y2, x3, y3, x4, y4 = x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
            if img_name not in file_label_dict:
                file_label_dict[img_name] = []
            file_label_dict[img_name].append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                              "x3": x3, "y3": y3, "x4": x4, "y4": y4, "word": word})
    print("读取XML标注文件完成：{0}个文件".format(len(file_label_dict)))

    qry_words_count_dict = {}
    for qry_w in qry_words:
        qry_words_count_dict[qry_w] = 0
    all_words_f = open(all_words_file, 'w', encoding='utf-8')
    print("生成txt文件...")
    for img_name in tqdm.tqdm(sorted(os.listdir(img_path))):
        img_name = str(img_name).split('.')[0]
        if is_vis:
            im = cv2.imread(os.path.join(img_path, "{0}.jpg".format(img_name)))
        with open(os.path.join(label_path, "{0}.txt".format(img_name)), 'w', encoding='utf-8') as f:
            for v in file_label_dict[img_name]:
                all_words_f.write("{0}\n".format(v["word"]))
                if v["word"] in qry_words:
                    qry_words_count_dict[v["word"]] += 1
                f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n"
                        .format(v["x1"], v["y1"], v["x2"], v["y2"], v["x3"], v["y3"], v["x4"], v["y4"], v["word"]))
                if is_vis:
                    cv2.rectangle(im, (v["x1"], v["y1"]), (v["x3"], v["y3"]), (60, 20, 220), 2)
                    cv2.putText(im, v["word"], (v["x1"] - 10, v["y1"] - 10), font, 1, (0, 69, 255), 1)
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
