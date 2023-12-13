# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:42:05 2023

@author: user
"""

import os 
import numpy as np
import cv2
from PIL import Image


class database_module(object):
    def __init__(
                self,
                image_source_dir,
                database_file,
                batch,
                input_size,
                shuffle = False
            ):
        
        self.image_source_dir = image_source_dir
        self.database_file = database_file
        self.batch = batch
        self.input_size = input_size
        self.shuffle = shuffle

        self.load_data_list()
        
    def load_data_list(self):
        with open(self.database_file,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
            
        self.data_paths = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]
        self.data_labels2 = [int(x.split(' ')[3]) for x in lines] # family
        self.data_labels3 = [int(x.split(' ')[4]) for x in lines] # genus
        self.data_labels1 = [int(x.split(' ')[5]) for x in lines] # species
        self.data_labels4 = [int(x.split(' ')[1]) for x in lines] # class
        self.data_labels5 = [int(x.split(' ')[2]) for x in lines] # order
        self.data_num = len(self.data_paths)
        self.data_idx = np.arange(self.data_num)
        self.cursor = 0
        self.epoch = 0
        self.reset_data_list()
        
    def shuffle_data_list(self):
        np.random.shuffle(self.data_idx)
    
    def reset_data_list(self):
        if self.shuffle:
            print('shuffling')
            print(self.data_idx[0:10])            
            np.random.shuffle(self.data_idx)
            print(self.data_idx[0:10])
        self.cursor = 0
        
    def read_batch(self):
        img = []
        lbl1 = []
        lbl2 = []
        lbl3 = []
        lbl4 = []
        lbl5 = []
        while len(img) < self.batch:
            try:
              
                im = cv2.imread(self.data_paths[self.data_idx[self.cursor]])
                if im is None:
                   im = cv2.cvtColor(np.asarray(Image.open(self.data_paths[self.data_idx[self.cursor]]).convert('RGB')),cv2.COLOR_RGB2BGR)
                im = cv2.resize(im,(self.input_size[0:2]))
                if np.ndim(im) == 2:
                    img.append(cv2.cvtColor(im,cv2.COLOR_GRAY2RGB))
                else:
                    img.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                lbl1.append(self.data_labels1[self.data_idx[self.cursor]])
                lbl2.append(self.data_labels2[self.data_idx[self.cursor]])
                lbl3.append(self.data_labels3[self.data_idx[self.cursor]])
                lbl4.append(self.data_labels4[self.data_idx[self.cursor]])
                lbl5.append(self.data_labels5[self.data_idx[self.cursor]])
            except:
                pass
            
            self.cursor += 1
            if self.cursor >= self.data_num:
                self.reset_data_list()
                self.epoch += 1
        
        img = np.asarray(img,dtype=np.float32)/255.0
        lbl1 = np.asarray(lbl1,dtype=np.int32)
        lbl2 = np.asarray(lbl2,dtype=np.int32)
        lbl3 = np.asarray(lbl3,dtype=np.int32)
        lbl4 = np.asarray(lbl4,dtype=np.int32)
        lbl5 = np.asarray(lbl5,dtype=np.int32)
        return (img,lbl1,lbl2,lbl3,lbl4,lbl5)






































