# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:31:21 2023

@author: user
"""


from network_module_baseline import inception_resnetv2_module as incept_resv2_net 

image_dir_parent_train = "path/to/dataset/PlantClef2022"
image_dir_parent_test = "path/to/dataset/PlantClef2022"
train_file = "lists/momo_train.txt"
test_file =  "lists/momo_validation.txt"

checkpoint_model = "path/to/pretrained_models/slim_models/inception_resnet_v2_2016_08_30/inception_resnet_v2_2016_08_30.ckpt"
checkpoint_save_dir = "dir/to/save/checkpoints"



batch = 32
input_size = (299,299,3)
numclass = 8
numorder = 78
numfamily = 357
numgenus = 3210
numclasses = 10000
learning_rate = 0.0001
iterbatch = 4
max_iter = 5000000
val_freq = 500
val_iter = 74



network = incept_resv2_net(
        batch = batch,
        iterbatch = iterbatch,
        numclass = numclass,
        numorder = numorder,
        numfamily = numfamily,
        numgenus = numgenus,
        numclasses = numclasses,
        input_size = input_size,
        image_dir_parent_train = image_dir_parent_train,
        image_dir_parent_test = image_dir_parent_test,
        train_file = train_file,
        test_file = test_file,
        checkpoint_model = checkpoint_model,
        save_dir = checkpoint_save_dir,
        learning_rate = learning_rate,
        max_iter = max_iter,
        val_freq = val_freq,
        val_iter = val_iter
        )
