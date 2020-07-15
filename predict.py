#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import shutil
import pathlib2
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from post_processing import lanms
# from post_processing import locality_aware_nms as nms_locality

from config import global_cfg
from datasets import get_rotate_mat, CustomDataSetRBox
from model import WordRetrievalModel
from utils import setup_logger, cal_recall_precison_f1, cal_map, cal_overlap


def resize_img(im, long_size=2048, is_scale=True):
    """ resize image to be divisible by 32 """
    w, h = im.size
    resize_w, resize_h = w, h

    if is_scale:
        scale = long_size * 1.0 / max(w, h)
        resize_w, resize_h = int(resize_w * scale), int(resize_h * scale)

    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    im = im.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_w, ratio_h,  = resize_w / w, resize_h / h

    return im, ratio_w, ratio_h


def is_valid_poly(res, score_shape, scale):
    """ check if the poly in image scope
        Input:
            res        : restored poly in original image
            score_shape: score map shape
            scale      : feature map -> image
        Output:
            True if valid
    """
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_poly(valid_pos, valid_geo, score_shape, scale=4):
    """ restore polys from feature maps in given positions
        Input:
            valid_pos  : potential text positions <numpy.ndarray, (n,2)>
            valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
            score_shape: shape of score map
            scale      : image / feature map
        Output:
            restored polys <numpy.ndarray, (n,8)>, index
    """
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x, y = valid_pos[i, 0], valid_pos[i, 1]
        y_min, y_max = y - d[0, i], y + d[1, i]
        x_min, x_max = x - d[2, i], x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x, temp_y = np.array([[x_min, x_max, x_max, x_min]]) - x, np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, cls_score_thresh=0.9, bbox_nms_overlap=0.4):
    """ get boxes from feature map
        Input:
            score       : score map from model <numpy.ndarray, (1,row,col)>
            geo         : geo map from model <numpy.ndarray, (5,row,col)>
            cls_score_thresh: threshold to segment score map
            bbox_nms_overlap  : threshold in nms
        Output:
            boxes       : final polys <numpy.ndarray, (n,9)>
    """
    score = score[0, :, :]
    xy_text = np.argwhere(score > cls_score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None, None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_poly(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None, None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes, keep = lanms.merge_quadrangle_n9(boxes.astype('float32'), bbox_nms_overlap)
    # boxes, keep = nms_locality.nms_locality(boxes.astype(np.float64), bbox_nms_overlap)
    return boxes, keep


def adjust_ratio(boxes, ratio_w, ratio_h):
    """ refine boxes
        Input:
            boxes  : detected polys <numpy.ndarray, (n,9)>
            ratio_w: ratio of width
            ratio_h: ratio of height
        Output:
            refined boxes
    """
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


class Predictor:
    def __init__(self, config, gpu_id=None):
        self.config = config
        self.config['tester']['output_dir'] = os.path.join(str(pathlib2.Path(os.path.abspath(__name__)).parent), self.config['tester']['output_dir'])
        self.data_cfg = self.config["data_cfg"]
        self.config["arch"]["pre_trained"] = False
        self.dataset_name = self.data_cfg['name']
        self.method_name = "{0}_{1}".format(self.config['arch']['backbone'], self.dataset_name)
        self.save_dir = os.path.join(self.config['tester']['output_dir'], self.method_name)
        self.model_path = os.path.join(self.save_dir, 'checkpoint', 'WordRetrievalNet_best.pth')
        self.logger = setup_logger(os.path.join(self.save_dir, 'predict_log'))
        self.label_encoder = LabelEncoder()
        self.qbs_res_dir = os.path.join(self.save_dir, "QbS_res")
        if not os.path.exists(self.qbs_res_dir):
            os.makedirs(self.qbs_res_dir)
        self.qbs_word_res_dir = os.path.join(self.save_dir, "QbS_word_res")
        if not os.path.exists(self.qbs_word_res_dir):
            os.makedirs(self.qbs_word_res_dir)

        # GPU ID
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.logger.info('Device {}'.format(self.device))

        # eval args
        self.long_size = self.config['tester']['long_size']
        self.cls_score_thresh = self.config['tester']['cls_score_thresh']
        self.bbox_nms_overlap = self.config['tester']['bbox_nms_overlap']
        self.query_nms_overlap = self.config['tester']['query_nms_overlap']
        self.overlap_thresh = self.config['tester']['overlap_thresh']
        self.metric = self.config['tester']['distance_metric']

        self.logger.info('Loading all data...')
        self.data_set = CustomDataSetRBox(self.data_cfg, max_img_length=self.config["trainer"]["input_size"],
                                          long_size=self.config['trainer']['long_size'])
        self.embedding_size = self.data_set.embedding_size
        self.words_embeddings = self.data_set.words_embeddings

        # test data path
        self.test_img_path = self.data_set.test_img_path
        self.test_gt_path = self.data_set.test_gt_path
        self.test_img_files = self.data_set.test_img_files
        self.test_gt_files = self.data_set.test_gt_files
        self.test_words = self.data_set.test_words
        self.train_unique_words = self.data_set.train_unique_words
        self.npy_path = os.path.join(self.config['tester']['output_dir'], 'predict_result_{0}.npy'.format(self.method_name))

    def predict(self):
        self.logger.info('Loading model and weights...')
        self.net = WordRetrievalModel(n_out=self.embedding_size, backbone=self.config["arch"]["backbone"],
                                      pre_trained=self.config["arch"]["pre_trained"])
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        self.logger.info("Load checkpoint from {}".format(self.model_path))
        self.net.load_state_dict({k.replace('module.', '').replace('phoc_output.', 'embedding_output.')
                                 .replace('rbox_output.', 'bbox_output.'): v for k, v in self.checkpoint['state_dict'].items()})
        self.net.to(self.device)
        self.net.eval()

        self.logger.info('Start predicting...')
        pred_result = {"pred_coord": [], "pred_embedding": [], "for_cal_map": {}}
        for_cal_map = pred_result["for_cal_map"]
        for i, (img_file, gt_file) in enumerate(zip(self.test_img_files, self.test_gt_files)):
            self.logger.info('Idx: {0} ===> Get gt boxes & gt words & img...'.format(i))
            img_name = str(os.path.basename(img_file))
            pred_embedding_list, gt_boxes_list, gt_words_list = [], [], []
            with open(gt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().rstrip('\n').lstrip('\ufeff').strip().split(',', maxsplit=8)
                gt_boxes_list.append([int(ver) for ver in line[:8]])
                gt_words_list.append(str(line[-1]).strip().lower())

            im = Image.open(img_file)
            im = im.convert("RGB")
            im, ratio_w, ratio_h = resize_img(im, long_size=self.long_size)
            with torch.no_grad():
                if str(self.device).__contains__('cuda'):
                    torch.cuda.synchronize(self.device)
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                im = transform(im).unsqueeze(0)
                im = im.to(self.device)
                (predict_score, predict_geo), predict_embed = self.net(im)
                if str(self.device).__contains__('cuda'):
                    torch.cuda.synchronize(self.device)
            predict_boxes, _ = get_boxes(score=predict_score.squeeze(0).cpu().numpy(), geo=predict_geo.squeeze(0).cpu().numpy(),
                                         cls_score_thresh=self.cls_score_thresh, bbox_nms_overlap=self.bbox_nms_overlap)
            predict_embed = predict_embed.squeeze(0).cpu().numpy()
            self.logger.info('Idx: {0} ===> Predict finish...'.format(i))

            if predict_boxes is None:
                continue
            self.logger.info('Idx: {0} ===> Predict result [predict_boxes: {1}; gt_boxes: {2}]'
                             .format(i, predict_boxes.shape, len(gt_boxes_list)))
            for predict_box in predict_boxes:
                min_x = min(predict_box[0], predict_box[2], predict_box[4], predict_box[6])
                max_x = max(predict_box[0], predict_box[2], predict_box[4], predict_box[6])
                min_y = min(predict_box[1], predict_box[3], predict_box[5], predict_box[7])
                max_y = max(predict_box[1], predict_box[3], predict_box[5], predict_box[7])
                w, h = max_x - min_x, max_y - min_y
                differ = h * 0.2 if h < w else w * 0.2
                min_x, max_x = int((min_x + differ) / 4), int((max_x - differ) / 4)
                min_y, max_y = int((min_y + differ) / 4), int((max_y - differ) / 4)
                if min_x > max_x or min_y > max_y:
                    continue
                pred_embedding_list.append(np.mean(predict_embed[:, min_y:max_y, min_x:max_x], axis=(1, 2)))
            predict_boxes = adjust_ratio(predict_boxes, ratio_w, ratio_h)
            for box in predict_boxes:
                pred_result["pred_coord"].append({"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                                                  "x3": box[4], "y3": box[5], "x4": box[6], "y4": box[7], "img_name": img_name})
            pred_result["pred_embedding"].extend(pred_embedding_list)

            for_cal_map[img_name] = {"pred_coord": predict_boxes, "pred_embedding": pred_embedding_list,
                                     "gt_box": gt_boxes_list, "gt_word": gt_words_list}

        np.save(self.npy_path, pred_result)
        self.logger.info("Save predict result to {0}".format(self.npy_path))

    def cal_f1(self):
        self.logger.info('Calculate Recall Precision F1...')
        result_save_path = os.path.join(self.save_dir, 'result')
        if os.path.exists(result_save_path):
            shutil.rmtree(result_save_path, ignore_errors=True)
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
        self.logger.info("Load predict result from {}".format(self.npy_path))
        pred_result = np.load(self.npy_path).tolist()
        for_cal_map = pred_result["for_cal_map"]
        for i, img_name in enumerate(sorted(os.listdir(self.test_img_path))):
            predict_boxes = for_cal_map[img_name]['pred_coord']
            seq = []
            if predict_boxes is not None:
                seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in predict_boxes])
            with open(os.path.join(result_save_path, str(os.path.basename(img_name).split('.')[0]) + '.txt'), 'w') as f:
                f.writelines(seq)

        res_dict = cal_recall_precison_f1(gt_path=self.test_gt_path, result_path=result_save_path)
        precision, recall, hmean = res_dict['precision'], res_dict['recall'], res_dict['hmean']
        self.logger.info('iou=0.5 ##### precision: {:.6f}, recall: {:.6f}, f1: {:.6f}'.format(precision, recall, hmean))

    def cal_map(self):
        self.logger.info('Calculate MAP...')
        predict_embeddings, joint_boxes, all_gt_boxes = [], [], []
        qbs_words, qbs_queries, qbs_targets, db_targets, gt_targets = [], [], [], [], []
        overlaps, used_test_word = [], []

        self.logger.info('Compute a mapping from class string to class id...')
        self.label_encoder.fit([word for word in self.test_words])

        self.logger.info('Create queries...')
        test_unique_words, counts = np.unique(self.test_words, return_counts=True)
        for idx, test_word in enumerate(self.test_words):
            gt_targets.extend(self.label_encoder.transform([test_word]))
            if test_word not in used_test_word and test_word in test_unique_words:
                qbs_words.append(test_word)
                qbs_queries.append(self.words_embeddings[test_word])
                qbs_targets.extend(self.label_encoder.transform([test_word]))
                used_test_word.append(test_word)

        self.logger.info("Load predict result from {}".format(self.npy_path))
        pred_result = np.load(self.npy_path).tolist()
        for_cal_map = pred_result["for_cal_map"]
        predict_embeddings = []
        for i, img_name in enumerate(sorted(os.listdir(self.test_img_path))):
            predict_boxes = for_cal_map[img_name]['pred_coord']
            gt_boxes, gt_words = for_cal_map[img_name]["gt_box"], for_cal_map[img_name]["gt_word"]
            predict_embeddings.extend(for_cal_map[img_name]['pred_embedding'])

            joint_boxes.extend(predict_boxes[:, :8])
            all_gt_boxes.extend(gt_boxes)
            gt_boxes = np.array(gt_boxes)
            self.logger.info('Idx: {0} ===> Calculate overlap...'.format(i))
            overlap = cal_overlap(predict_boxes, gt_boxes)
            overlaps.append(overlap)
            inds = overlap.argmax(axis=1)
            db_targets.extend(self.label_encoder.transform([gt_words[idx] for idx in inds]))

        self.logger.info('End evaluate...')
        db = np.vstack(predict_embeddings)
        all_overlaps = np.zeros((len(joint_boxes), len(all_gt_boxes)), dtype=np.float32)
        x, y = 0, 0
        for o in overlaps:
            all_overlaps[y:y + o.shape[0], x: x + o.shape[1]] = o
            y += o.shape[0]
            x += o.shape[1]
        db_targets = np.array(db_targets)
        qbs_targets = np.array(qbs_targets)
        qbs_words = np.array(qbs_words)
        qbs_queries = np.array(qbs_queries)
        joint_boxes = np.array(joint_boxes)

        assert (qbs_queries.shape[0] == qbs_targets.shape[0])
        assert (db.shape[0] == db_targets.shape[0])

        self.logger.info('Calculate mAP...')
        for over_thresh in self.overlap_thresh:
            mAP_qbs, mR_qbs = cal_map(qbs_queries, qbs_targets, db, db_targets, gt_targets, joint_boxes,
                                      all_overlaps, self.query_nms_overlap, over_thresh, qbs_words, num_workers=24)
            mAP_qbs, mR_qbs = np.mean(mAP_qbs * 100), np.mean(mR_qbs * 100)
            self.logger.info('Overlap_thresh = %3.2f ###### QbS mAP: %3.2f, mR: %3.2f' % (over_thresh, mAP_qbs, mR_qbs))

    def query_word_string(self, query_string, topN):
        QbS_res = {}
        self.logger.info("Load predict result from {}".format(self.npy_path))
        pred_result = np.load(self.npy_path).tolist()
        pred_coord_list, pred_embedding_list = pred_result["pred_coord"], pred_result["pred_embedding"]
        word_idx = 0
        if query_string in self.words_embeddings.keys():
            query_embedding = np.array([self.words_embeddings[query_string]])
        else:
            raise ValueError("query not in word_str list")
        dist_mat = cdist(XA=query_embedding, XB=pred_embedding_list, metric="cosine")
        retrieval_indices = np.argsort(dist_mat, axis=1)[0].tolist()
        retrieval_indices = retrieval_indices[:topN]
        retrieval_coordinate_list = [pred_coord_list[ind] for ind in retrieval_indices]
        for coord, ind in zip(retrieval_coordinate_list, retrieval_indices):
            img_name, x1, y1, x2, y2, x3, y3, x4, y4, dis = coord["img_name"], \
                                                            coord["x1"], coord["y1"], coord["x2"], coord["y2"], \
                                                            coord["x3"], coord["y3"], coord["x4"], coord["y4"], \
                                                            dist_mat[0][ind]
            print("query_string: {0}; cont: {1}; cos dis: {2};".format(query_string, word_idx + 1, dis))
            if img_name not in QbS_res:
                QbS_res[img_name] = []
            QbS_res[img_name].append([x1, y1, x2, y2, x3, y3, x4, y4, dis, query_string, word_idx])
            word_idx += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        for key, value in QbS_res.items():
            im = cv2.imread(os.path.join(self.test_img_path, str(key)))
            for val in value:
                x_min, y_min, x_max, y_max, word = int(val[0]), int(val[1]), int(val[4]), int(val[5]), str(val[9])
                cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0, 255, 0), 6)
                if not os.path.exists(os.path.join(self.qbs_word_res_dir, word)):
                    os.makedirs(os.path.join(self.qbs_word_res_dir, word))
                cv2.imwrite(os.path.join(self.qbs_word_res_dir, word, "{0}_{1}.png".format(word, str(val[10]))),
                            im[y_min: y_max, x_min: x_max])
                cv2.putText(im, "{0} {1}".format(word, str(val[8])), (x_min - 10, y_min - 10), font, 1.2, (60, 20, 220), 2)
            cv2.imwrite(os.path.join(self.qbs_res_dir, str(key)), im)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    predictor = Predictor(global_cfg, gpu_id=1)
    predictor.predict()  # generate npy file
    predictor.cal_f1()   # calculate Recall Precision F1
    predictor.cal_map()  # calculate MAP
    predictor.query_word_string("de", 10)  # query word string defined by users
