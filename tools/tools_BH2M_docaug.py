#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

__author__ = 'peng zhao'

import os
import shutil
import numpy as np
import skimage.filters as fi
import skimage.transform as tf
import skimage.morphology as mor
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu


class Cfg:
    pre_path = "/home/zhaopeng/WordSpottingDatasets/"
    data_name = "BH2M"
    img_path = os.path.join(pre_path, data_name, "train", "images")
    label_path = os.path.join(pre_path, data_name, "train", "labels")
    gen_img_path = os.path.join(pre_path, data_name, "gen", "images")
    gen_label_path = os.path.join(pre_path, data_name, "gen", "labels")


# ### 增广单词图像 ####
def aug_crop(img, tparams):
    t_img = img < threshold_otsu(img)
    nz = t_img.nonzero()
    h_pad = np.random.randint(low=tparams['hpad'][0], high=tparams['hpad'][1], size=2)
    v_pad = np.random.randint(low=tparams['vpad'][0], high=tparams['vpad'][1], size=2)
    b = [max(nz[1].min() - h_pad[0], 0), max(nz[0].min() - v_pad[0], 0),
         min(nz[1].max() + h_pad[1], img.shape[1]), min(nz[0].max() + v_pad[1], img.shape[0])]
    return img[b[1]:b[3], b[0]:b[2]]


def affine(img, tparams):
    phi = (np.random.uniform(tparams['shear'][0], tparams['shear'][1]) / 180) * np.pi
    theta = (np.random.uniform(tparams['rotate'][0], tparams['rotate'][1]) / 180) * np.pi
    t = tf.AffineTransform(shear=phi, rotation=theta, translation=(-25, -50))
    tmp = tf.warp(img, t, order=tparams['order'], mode='edge', output_shape=(img.shape[0] + 100, img.shape[1] + 100))
    return tmp


def morph(img, tparams):
    ops = [mor.grey.erosion, mor.grey.dilation]
    t = ops[np.random.randint(2)]
    selem = mor.square(np.random.randint(1, (tparams['selem_size'][0] if t == 0 else tparams['selem_size'][1])))
    return t(img, selem)


def augment(word, tparams, keep_size=False):
    assert (word.ndim == 2)
    t = np.zeros_like(word)
    s = np.array(word.shape) - 1
    t[0, :], t[:, 0] = word[0, :], word[:, 0]
    t[s[0], :], t[:, s[1]] = word[s[0], :], word[:, s[1]]
    pad = np.median(t[t > 0])

    tmp = np.ones((word.shape[0] + 8, word.shape[1] + 8), dtype=word.dtype) * pad
    tmp[4:-4, 4:-4] = word
    out = tmp
    out = affine(out, tparams)
    out = aug_crop(out, tparams)
    out = morph(out, tparams)
    if keep_size:
        out = tf.resize(out, word.shape)
    out = np.round(out).astype(np.ubyte)
    return out


# ### 加载数据集 ####
def load_dataset():
    print("读取标注文件...")
    file_label_dict = {}
    for label_name in os.listdir(Cfg.label_path):
        with open(os.path.join(Cfg.label_path, label_name), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            img_name = str(label_name).split('.')[0]
            line = line.strip().rstrip('\n').lstrip('\ufeff').strip().split(',', maxsplit=8)
            x1, y1, x2, y2, x3, y3, x4, y4 = [int(ver) for ver in line[:8]]
            word = str(line[-1]).strip()
            if img_name not in file_label_dict:
                file_label_dict[img_name] = []
            file_label_dict[img_name].append({"x1": x1, "y1": y1, "x2": x3, "y2": y3, "word": word})
    print("读取标注文件完成：{0}个文件".format(len(file_label_dict)))

    data = []
    for img_name in sorted(os.listdir(Cfg.img_path)):
        img_name = str(img_name).split('.')[0]
        box_id = 0
        gt_boxes, regions = [], []
        for v in file_label_dict[img_name]:
            gt_boxes.append([v["x1"], v["y1"], v["x2"], v["y2"]])
            regions.append({'id': box_id, 'image': img_name, 'height': v["y2"] - v["y1"], 'width': v["x2"] - v["x1"],
                            'label': v["word"], 'x': v["x1"], 'y': v["y1"]})
            box_id += 1
        data.append({'id': img_name, 'gt_boxes': gt_boxes, 'regions': regions})
    return data


def close_crop_box(im, box):
    gray = rgb2gray(im[box[1]:box[3], box[0]:box[2]])
    t_img = gray < threshold_otsu(gray)
    h_proj, v_proj = t_img.sum(axis=0), t_img.sum(axis=1)
    x1o = box[0] + max(h_proj.nonzero()[0][0] - 1, 0)
    y1o = box[1] + max(v_proj.nonzero()[0][0] - 1, 0)
    x2o = box[2] - max(h_proj.shape[0] - h_proj.nonzero()[0].max() - 1, 0)
    y2o = box[3] - max(v_proj.shape[0] - v_proj.nonzero()[0].max() - 1, 0)
    obox = (x1o, y1o, x2o, y2o)
    return obox


# ### 就地数据增广 ####
def inplace_augment(data, tparams=None):
    if tparams is None:
        tparams = {}
        tparams['samples_per_image'] = 5
        tparams['shear'] = (-5, 30)
        tparams['order'] = 1  # bilinear
        tparams['selem_size'] = (3, 4)  # max size for square selem for erosion, dilation

    tparams['rotate'] = (0, 1)
    tparams['hpad'] = (0, 1)
    tparams['vpad'] = (0, 1)

    for i, datum in enumerate(data):
        img_name = datum['id']
        for j in range(tparams['samples_per_image']):
            im = imread(os.path.join(Cfg.img_path, "{0}.jpg".format(img_name)))
            if im.ndim == 3:
                im = img_as_ubyte(rgb2gray(im))

            out = im.copy()
            for jj, b in enumerate(reversed(datum['gt_boxes'])):
                try:  # Some random values for weird boxes give value errors, just handle and ignore
                    b = close_crop_box(im, b)
                    word = im[b[1]:b[3], b[0]:b[2]]
                    aug = augment(word, tparams, keep_size=True)
                except ValueError:
                    continue
                out[b[1]:b[3], b[0]:b[2]] = aug

            imsave(os.path.join(Cfg.gen_img_path, '{0}_{1}.jpg'.format(img_name, j)), out)
            shutil.copyfile(os.path.join(Cfg.label_path, "{0}.txt".format(img_name)),
                             os.path.join(Cfg.gen_label_path, '{0}_{1}.txt'.format(img_name, j)))
            print("inplace_augment===> {0} ===> {1}_{2}.txt".format(i, img_name, j))


def build_vocab(data):
    """ Builds a set that contains the vocab. Filters infrequent tokens. """
    texts = []
    for datum in data:
        for r in datum['regions']:
            texts.append(r['label'])
    vocab, index = np.unique(texts, return_index=True)
    return vocab, index


def create_background(m, shape, fstd=2, bstd=10):
    canvas = np.ones(shape) * m
    noise = np.random.randn(shape[0], shape[1]) * bstd
    noise = fi.gaussian(noise, fstd)  # low-pass filter noise
    canvas += noise
    canvas = np.round(canvas).astype(np.uint8)
    return canvas


# ### 全页数据增广 ####
def fullpage_augment(data, num_images=1000, is_augment=True):
    vocab, _ = build_vocab(data)
    vocab_size = len(vocab)
    wtoi = {w: i for i, w in enumerate(vocab)}

    tparams = {}
    tparams['shear'] = (-5, 30)
    tparams['order'] = 1  # bilinear
    tparams['selem_size'] = (3, 4)  # max size for square selem for erosion, dilation
    tparams['rotate'] = (0, 1)
    tparams['hpad'] = (0, 12)
    tparams['vpad'] = (0, 12)

    words_by_label = [[] for i in range(vocab_size)]
    shapes, medians = [], []
    for datum in data:
        im = imread(os.path.join(Cfg.img_path, "{0}.jpg".format(datum['id'])))
        if im.ndim == 3:
            im = img_as_ubyte(rgb2gray(im))

        shapes.append(im.shape)
        medians.append(np.median(im))
        for r in datum['regions']:
            x1, y1, x2, y2 = r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height']
            word, label = im[y1:y2, x1:x2], r['label']
            ind = wtoi[label]
            words_by_label[ind].append(word)

    m = int(np.median(medians))
    nwords = 256  # batch size?
    s = 3  # inter word space
    box_id = 0
    for i in range(num_images):
        shape = shapes[i % len(shapes)]
        canvas = create_background(m + np.random.randint(0, 20) - 10, shape)
        x, y = int(shape[1] * 0.08), int(shape[0] * 0.08)  # Upper left corner of box
        maxy = 0
        f = os.path.join(Cfg.gen_img_path, "fullpage_{0}.jpg".format(i))
        f_txt = open(os.path.join(Cfg.gen_label_path, "fullpage_{0}.txt".format(i)), mode='w', encoding='utf-8')
        for j in range(nwords):
            ind = np.random.randint(vocab_size)
            k = len(words_by_label[ind])
            word = words_by_label[ind][np.random.randint(k)]
            # randomly transform word and place on canvas
            if is_augment:
                try:
                    tword = augment(word, tparams)
                except:
                    tword = word
            else:
                tword = word

            h, w = tword.shape
            if x + w > int(shape[1] * 0.92):  # done with row?
                x, y = int(shape[1] * 0.08), maxy + s
            if y + h > int(shape[0] * 0.92):  # done with page?
                break

            x1, y1, x2, y2 = x, y, x + w, y + h
            canvas[y1:y2, x1:x2] = tword
            b = [x1, y1, x2, y2]
            x = x2 + s
            maxy = max(maxy, y2)
            f_txt.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3], vocab[ind]))
            box_id += 1
        imsave(f, canvas)
        f_txt.close()
        print("fullpage_augment===> {0} ===> {1}".format(i, f))


if __name__ == "__main__":
    data = load_dataset()

    gen_images = 2000
    num_data = len(data)
    tparams = {}
    # get approximately the same amount of images
    tparams['samples_per_image'] = int(np.round(float(gen_images / 2) / num_data))
    tparams['shear'] = (-5, 30)
    tparams['order'] = 1  # bilinear
    tparams['selem_size'] = (3, 4)  # max size for square kernel for erosion, dilation
    inplace_augment(data, tparams=tparams)

    nps = gen_images - tparams['samples_per_image'] * num_data
    fullpage_augment(data, nps)
