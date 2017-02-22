#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
import tensorflow.contrib.slim as slim
import tensorflow as tf
import config as cfg
# import numpy as np
# import cv2

batch_size = cfg.BATCH_SIZE
image_height = cfg.IMAGE_HEIGHT
image_width = cfg.IMAGE_WIDTH
image_channels = cfg.IMAGE_CHANNELS
flo_channels = cfg.FLO_CHANNELS


def inference(img1, img2):
    concat1 = tf.concat(3, [img1, img2], name='concat1')
    conv1 = slim.conv2d(concat1, 64, [7, 7], 2, 'SAME', scope='conv1')
    conv2 = slim.conv2d(conv1, 128, [5, 5], 2, 'SAME', scope='conv2')
    conv3 = slim.conv2d(conv2, 256, [5, 5], 2, 'SAME', scope='conv3')
    conv3_1 = slim.conv2d(conv3, 256, [3, 3], 1, 'SAME', scope='conv3_1')
    conv4 = slim.conv2d(conv3_1, 512, [3, 3], 2, 'SAME', scope='conv4')
    conv4_1 = slim.conv2d(conv4, 512, [3, 3], 1, 'SAME', scope='conv4_1')
    conv5 = slim.conv2d(conv4_1, 512, [3, 3], 2, 'SAME', scope='conv5')
    conv5_1 = slim.conv2d(conv5, 512, [3, 3], 1, 'SAME', scope='conv5_1')
    conv6 = slim.conv2d(conv5_1, 1024, [3, 3], 2, 'SAME', scope='conv6')
    conv6_1 = slim.conv2d(conv6, 1024, [3, 3], 1, 'SAME', scope='conv6_1')
    # 6 * 8 flow
    predict6 = slim.conv2d(conv6_1, 2, [3, 3], 1, 'SAME', scope='predict6')
    # 12 * 16 flow
    deconv5 = slim.conv2d_transpose(conv6_1, 512, [4, 4], 2, 'SAME', scope='deconv5')
    deconvflow6 = slim.conv2d_transpose(predict6, 2, [4, 4], 2, 'SAME', scope='deconvflow6')
    concat5 = tf.concat(3, [conv5_1, deconv5, deconvflow6], name='concat5')
    predict5 = slim.conv2d(concat5, 2, [3, 3], 1, 'SAME', scope='predict5')
    # 24 * 32 flow
    deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], 2, 'SAME', scope='deconv4')
    deconvflow5 = slim.conv2d_transpose(predict5, 2, [4, 4], 2, 'SAME', scope='deconvflow5')
    concat4 = tf.concat(3, [conv4_1, deconv4, deconvflow5], name='concat4')
    predict4 = slim.conv2d(concat4, 2, [3, 3], 1, 'SAME', scope='predict4')
    # 48 * 64 flow
    deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], 2, 'SAME', scope='deconv3')
    deconvflow4 = slim.conv2d_transpose(predict4, 2, [4, 4], 2, 'SAME', scope='deconvflow4')
    concat3 = tf.concat(3, [conv3_1, deconv3, deconvflow4], name='concat3')
    predict3 = slim.conv2d(concat3, 2, [3, 3], 1, 'SAME', scope='predict3')
    # 96 * 128 flow
    deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], 2, 'SAME', scope='deconv2')
    deconvflow3 = slim.conv2d_transpose(predict3, 2, [4, 4], 2, 'SAME', scope='deconvflow3')
    concat2 = tf.concat(3, [conv2, deconv2, deconvflow3], name='concat2')
    predict2 = slim.conv2d(concat2, 2, [3, 3], 1, 'SAME', scope='predict2')
    return predict6, predict5, predict4, predict3, predict2


def l1_loss(tensor, name):
    loss = tf.reduce_sum(tf.abs(tensor), name=name)
    tf.add_to_collection('losses', loss)
    tf.summary.scalar('losses' + name, loss)
    return loss


def loss(predict6, predict5, predict4, predict3, predict2, flo):
    flo6 = tf.image.resize_images(flo, [6, 8])
    loss6 = 0.32 * l1_loss(flo6 - predict6, 'loss6')
    flo5 = tf.image.resize_images(flo, [12, 16])
    loss5 = 0.08 * l1_loss(flo5 - predict5, 'loss5')
    flo4 = tf.image.resize_images(flo, [24, 32])
    loss4 = 0.02 * l1_loss(flo4 - predict4, 'loss4')
    flo3 = tf.image.resize_images(flo, [48, 64])
    loss3 = 0.01 * l1_loss(flo3 - predict3, 'loss3')
    flo2 = tf.image.resize_images(flo, [96, 128])
    loss2 = 0.005 * l1_loss(flo2 - predict2, 'loss2')
    total_loss = tf.add_n([loss6, loss5, loss4, loss3, loss2], name='total_loss')
    tf.summary.scalar('losses' + 'total_loss', total_loss)
    return total_loss


def placeholder_inputs():
    img1_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, image_channels))
    img2_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, image_channels))
    flo_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, flo_channels))
    return img1_placeholder, img2_placeholder, flo_placeholder


def fill_feed_dict(data, img1_pl, img2_pl, flo_pl):
    img1_feed, img2_feed, flo_feed = data.get_batch(batch_size)
    # test data
    # cv2.imshow('image1', img1_feed[0].astype(np.uint8))
    # cv2.imshow('image2', img2_feed[0].astype(np.uint8))
    # cv2.waitKey()
    feed_dict = {img1_pl: img1_feed, img2_pl: img2_feed, flo_pl: flo_feed}
    return feed_dict
