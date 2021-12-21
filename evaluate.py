# -*- coding: utf-8 -*-
import tensorflow as tf
import tiny_faces_model as tiny_model
import util
from argparse import ArgumentParser
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import cv2
import pickle

# import pylab as pl
import time
import os
import sys
from scipy.special import expit
import glob

MAX_INPUT_DIM = 5000.0

def evaluate(weight_file_path, output_dir=None, data_dir=None, img=None, list_imgs=None,
             prob_thresh=0.5, nms_thresh=0.1, lw=3, display=False,
             draw=True, save=True, print_=0):
    """
  Detect faces in images.
  :param weight_file_path: A pretrained weight file in the pickle format 
        generated by matconvnet_hr101_to_tf.py.
  :param output_dir: A directory into which images with detected faces are output.
        default=None to not output detected faces
  :param data_dir: A directory which contains images for face detection.
  :param img: One image for face detection 
  :param prob_thresh: The threshold of detection confidence.
  :param nms_thresh: The overlap threshold of non maximum suppression
  :param lw: Line width of bounding boxes. If zero specified,
        this is determined based on confidence of each detection.
  :param display: Display tiny face images on window.
  :param draw: Draw bouding boxes on images.
  :param save: Save images in output_dir.
  :param print_: 0 for no print, 1 for light print, 2 for full print
  :return: final bboxes
  """
    if type(img) != np.ndarray:
        one_pic = False
    else:
        one_pic = True

    if not output_dir:
        save = False
        draw = False

    # list of bounding boxes for the pictures
    final_bboxes = []

    # placeholder of input images. Currently batch size of one is supported.
    x = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])  # n, h, w, c

    # Create the tiny face model which weights are loaded from a pretrained model.
    model = tiny_model.Model(weight_file_path)
    score_final = model.tiny_face(x)

    # Load an average image and clusters(reference boxes of templates).
    with open(weight_file_path, "rb") as f:
        _, mat_params_dict = pickle.load(f)

    # Average RGB values from model
    average_image = model.get_data_by_key("average_image")

    # Reference boxes of template for 05x, 1x, and 2x scale
    clusters = model.get_data_by_key("clusters") # 一些参考的包围盒，shape为[25, 5]。共有25组，每组最后一个是scale，前面四个是x, y, w+x, h+x
    clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    normal_idx = np.where(clusters[:, 4] == 1)

    # Find image files in data_dir.
    filenames = []
    # if we provide only one picture, no need to list files in dir
    if one_pic:
        filenames = [img]
    elif type(list_imgs) == list:
        filenames = list_imgs
    else:
        for ext in ('*.png', '*.gif', '*.jpg', '*.jpeg'):
            filenames.extend(glob.glob(os.path.join(data_dir, ext)))

    # main
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer()) # 初始化所有变量
        for filename in filenames:
            # if we provide only one picture, no need to list files in dir
            if not one_pic and type(list_imgs) != list:
                fname = filename.split(os.sep)[-1]
                raw_img = cv2.imread(filename)
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            else:
                fname = 'current_picture'
                raw_img = filename
            raw_img_f = raw_img.astype(np.float32)

            def _calc_scales():
                """
        Compute the different scales for detection
        :return: [2^X] with X depending on the input image
        """
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
                min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                                np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
                max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
                scales_down = pl.arange(min_scale, 0, 1.)
                scales_up = pl.arange(0.5, max_scale, 0.5)
                scales_pow = np.hstack((scales_down, scales_up))
                scales = np.power(2.0, scales_pow)
                return scales

            scales = _calc_scales()
            start = time.time()

            # initialize output
            bboxes = np.empty(shape=(0, 5))

            # process input at different scales
            for s in scales:
                if print_ == 2:
                    print("Processing {} at scale {:.4f}".format(fname, s))
                img : np.ndarray = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                img = img - average_image
                img = img[np.newaxis, :]

                # we don't run every template on every scale ids of templates to ignore
                tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
                ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

                # run through the net
                print(x)
                score_final_tf = sess.run(score_final, feed_dict={x: img})
                print("success")

                # collect scores
                score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
                prob_cls_tf = expit(score_cls_tf) #sigmoid
                prob_cls_tf[0, :, :, ignoredTids] = 0.0

                def _calc_bounding_boxes():
                    # threshold for detection
                    _, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)

                    # interpret heatmap into bounding boxes
                    cy = fy * 8 - 1
                    cx = fx * 8 - 1
                    ch = clusters[fc, 3] - clusters[fc, 1] + 1
                    cw = clusters[fc, 2] - clusters[fc, 0] + 1

                    # extract bounding box refinement
                    Nt = clusters.shape[0]
                    tx = score_reg_tf[0, :, :, 0:Nt]
                    ty = score_reg_tf[0, :, :, Nt:2 * Nt]
                    tw = score_reg_tf[0, :, :, 2 * Nt:3 * Nt]
                    th = score_reg_tf[0, :, :, 3 * Nt:4 * Nt]

                    # refine bounding boxes
                    dcx = cw * tx[fy, fx, fc]
                    dcy = ch * ty[fy, fx, fc]
                    rcx = cx + dcx
                    rcy = cy + dcy
                    rcw = cw * np.exp(tw[fy, fx, fc])
                    rch = ch * np.exp(th[fy, fx, fc])

                    scores = score_cls_tf[0, fy, fx, fc]
                    tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
                    tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
                    tmp_bboxes = tmp_bboxes.transpose()
                    return tmp_bboxes

                tmp_bboxes = _calc_bounding_boxes()
                bboxes = np.vstack((bboxes, tmp_bboxes))  # <class 'tuple'>: (5265, 5)

            if print_ >= 1:
                print("time {:.2f} secs for {}".format(time.time() - start, fname))

            # non maximum suppression
            refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                                      tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                                      max_output_size=bboxes.shape[0], iou_threshold=nms_thresh)
            refind_idx = sess.run(refind_idx)
            refined_bboxes = bboxes[refind_idx]

            # convert bbox coordinates to int
            # f_box = overlay_bounding_boxes(raw_img, refined_bboxes, lw, draw)
            f_box = []
            for r in refined_bboxes:
                temp = []
                for m1 in r[:4]:
                    temp.append(int(m1))
                f_box.append(temp)

            if display:
                # plt.axis('off')
                plt.imshow(raw_img)
                plt.show()

            if save:
                # save image with bounding boxes
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, fname), raw_img)

            final_bboxes.append(f_box)

    if len(final_bboxes) == 1:
        final_bboxes = final_bboxes[0]
    return final_bboxes
