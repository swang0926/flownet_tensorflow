#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import struct
import numpy as np
import cv2
import os
import config as cfg

list_path = cfg.LIST_PATH
file_path = cfg.FILE_PATH


# part1:get_DataSets
# create all image and flo data list
def get_list(list_txt):
    list_dir = os.path.join(list_path, list_txt)
    name_list = []
    with open(list_dir) as list_dir:
        for line in list_dir:
            # due to some error in txt files , we have to amend the 'line'
            line = line.rstrip()
            line = line.split(' ')[0]
            name = os.path.join(file_path, line)
            name_list.append(name)
    return name_list


def get_nameset():
    train_img1 = get_list('img1_train.txt')
    train_img2 = get_list('img2_train.txt')
    train_flo = get_list('flo_train.txt')
    val_img1 = get_list('img1_val.txt')
    val_img2 = get_list('img2_val.txt')
    val_flo = get_list('flo_val.txt')
    assert len(train_img1) == len(train_img2)
    assert len(train_img1) == len(train_flo)
    assert len(val_img1) == len(val_img2)
    assert len(val_img1) == len(val_flo)
    trainset = Data([train_img1, train_img2, train_flo])
    valset = Data([val_img1, val_img2, val_flo])
    return trainset, valset


# part2:get batch
class Data(object):
    def __init__(self, nameset):
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.num_examples = len(nameset[0])
        self.img1 = np.array(nameset[0])
        self.img2 = np.array(nameset[1])
        self.flo = np.array(nameset[2])

    def read_flo(self, floname):
        f = open(floname, "rb")
        data = f.read()
        f.close()
        width = struct.unpack('@i', data[4:8])[0]
        height = struct.unpack('@i', data[8:12])[0]
        flodata = np.zeros((height, width, 2))
        for i in range(width*height):
            data_u = struct.unpack('@f', data[12+8*i:16+8*i])[0]
            data_v = struct.unpack('@f', data[16+8*i:20+8*i])[0]
            n = int(i / width)
            k = np.mod(i, width)
            flodata[n, k, :] = [data_u, data_v]
        return flodata

    def read_img(self, imgname):
        image = cv2.imread(imgname)
        image = image.astype(np.float32)
        return image

    def create_batch(self, eval_img, name_list):
        array = []
        for name in name_list:
            if eval_img:
                array.append(self.read_img(name))
            else:
                array.append(self.read_flo(name))
        array = np.asarray(array)
        return array

    def get_batch(self, batch_size):
        # get data index depends on the length of batch_size
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            start = 0
            self.index_in_epoch = batch_size
            # start next epoch
            self.epochs_completed += 1
            # shuffle data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.img1 = self.img1[perm]
            self.img2 = self.img2[perm]
            self.flo = self.flo[perm]
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        # get batch
        batch_img1 = self.create_batch(True, self.img1[start:end])
        batch_img2 = self.create_batch(True, self.img2[start:end])
        batch_flo = self.create_batch(False, self.flo[start:end])
        return batch_img1, batch_img2, batch_flo
