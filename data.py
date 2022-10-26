import tensorflow as tf
import os,sys
from glob import glob
import cv2
import re
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))

class DataLoader:
    """
    eg:
    cls_dict = get_cls_dict()
    data_loader = DataLoader('../dataset_4digits/train','../dataset_4digits/data.txt',cls_dict,img_format='png')
    """
    def __init__(self, img_dir, txt_path, epochs=1, batch_size=1, shuffle_size=1, img_format='png'):
        """
        img_dir: directory of train/test

        """
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.img_format = img_format

    def _gen_(self):
        labels = []
        with open(self.txt_path) as reader:
            ln_nums = []
            for i,line in enumerate(reader.readlines()):
                ln_num = round(float(line.strip()),2)
                if i%4 == 3:
                    ln_nums.append(ln_num)
                    labels.append(ln_nums)
                    ln_nums = []
                else:
                    ln_nums.append(ln_num)

        for name in self.get_filenames():
            img = cv2.imread(name)
            im_id = int(re.findall('\d+',os.path.basename(name))[0])
            values = labels[im_id-1] # image begin from 1
            yield img, values

    def get_dataset(self): # for big dataset, for training
        img_shape = self.get_img_shape()
        dataset = tf.data.Dataset.from_generator(self._gen_,(tf.float32,tf.float32),(img_shape,[4]))
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(self.shuffle_size)
        return dataset

    def get_one_epoch(self): # for small dataset, for testing
        return self.get_dataset().take(self.get_steps_per_epoch())

    def get_filenames(self):
        assert os.path.exists(self.img_dir)
        img_names = glob(os.path.join(self.img_dir,'images','*.'+self.img_format))
        if len(img_names)==0:
            img_names = glob(os.path.join(self.img_dir,'*.'+self.img_format))
        return img_names

    def get_img_shape(self):
        filenames = self.get_filenames()
        assert len(filenames)
        img = cv2.imread(filenames[0])
        return img.shape

    def get_steps_per_epoch(self):
        filenames = self.get_filenames()
        assert len(filenames)
        return int(len(filenames)/self.batch_size)

if __name__=='__main__':
    pass