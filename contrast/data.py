# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2021/12/20-9:50 下午
'''
import torch
import io
from PIL import Image
import os,sys
import torchvision
import cv2
import re
import numpy as np
from torch.utils.data import Dataset


class Data_Loader(Dataset):
    def __init__(self, imgpath, labelpath):
        self.name = os.listdir(imgpath)
        self.img_dir = imgpath
        self.lab_dir = labelpath
        self.image_path = self.get_img()
        self.label = self.get_label()
        self.images_transf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])

    def get_label(self):
        f = open(self.lab_dir)
        label = []
        for line in f.readlines():
            label.append(float(line))
        f.close()
        label = np.array(label).astype(np.float32)
        label.resize([len(label)//4, 4])
        return label

    def get_img(self):
        self.name.sort(key=lambda x: int(x[:-4]))
        images = []
        for i in range(len(self.name)):
            images.append(self.name[i])
        return images

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        fn, label = self.image_path[index],self.label[index]
        fn = self.img_dir + '/' + fn
        image = Image.open(fn)
        image = self.images_transf(image)
        label = torch.from_numpy(label)
        label =  label.int()
        return image, label




if __name__=='__main__':
    pass