# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:40:37 2023

@author: user
"""

import sys
sys.path.append("path/to/tf_slim/models/research/slim")
import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
import numpy as np

from nets.inception_resnet_v2 import inception_resnet_v2
from nets.inception_resnet_v2 import inception_resnet_v2_arg_scope
from nets import inception_utils
import csv
import os
from six.moves import cPickle
import datetime

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

image_source_dir = "path/to/dataset/PlantClef2022"
testset_txt = "lists/momo_validation.txt"
model_checkpoint = "path/to/trained/checkpoint.ckpt"
prediction_dir = "dir/to/save/csv_output"

create_dir(prediction_dir)

prediction_class_csv = os.path.join(prediction_dir, "predictions_class.csv")
prediction_order_csv = os.path.join(prediction_dir, "predictions_order.csv")
prediction_family_csv = os.path.join(prediction_dir, "predictions_family.csv")
prediction_genus_csv = os.path.join(prediction_dir, "predictions_genus.csv")
prediction_species_csv = os.path.join(prediction_dir, "predictions_species.csv")



with open(testset_txt, 'r') as txt:
    lines = [x for x in txt.readlines()]
    
filepaths = [os.path.join(image_source_dir, x.rsplit(" ")[0]) for x in lines]
f_class = [x.rsplit(" ")[1] for x in lines]
f_order = [x.split(" ")[2] for x in lines]
f_family = [x.split(" ")[3] for x in lines]
f_genus = [x.split(" ")[4] for x in lines]
f_species = [x.split(" ")[5] for x in lines]



#   Parameters
global_batch = 6 
batch = 60
numclasses = 10000
input_size = (299,299,3)
topN = 5

 

#   Initiate tensors
x = tf.placeholder(tf.float32,(batch,) + input_size)
y = tf.placeholder(tf.int32,(batch,))
is_training = tf.placeholder(tf.bool)
is_training2 = tf.placeholder(tf.bool)
tf_filepath2 =  tf.placeholder(tf.string,shape=(global_batch,))

#   Image processing methods
train_preproc = lambda xi: inception_preprocessing.preprocess_image(
        xi,input_size[0],input_size[1],is_training=True)

test_preproc = lambda xi: inception_preprocessing.preprocess_image(
        xi,input_size[0],input_size[1],is_training=False)  


def datetimestr():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

def read_images(p):
    im = tf.io.read_file(p)    
    im =  tf.cast(tf.image.resize_images(tf.image.decode_jpeg(
        im, channels=3),(input_size[:2])),tf.float32)
    
    im1 = im[0:260,0:260,:]
    im2 = im[0:260,-260:,:]
    im3 = im[-260:,0:260,:]
    im4 = im[-260:,-260:,:]
    im5 = im[19:279,19:279,:]
    
    im1 =  tf.cast(tf.image.resize_images(im1,(299,299)),tf.float32)
    im2 =  tf.cast(tf.image.resize_images(im2,(299,299)),tf.float32)
    im3 =  tf.cast(tf.image.resize_images(im3,(299,299)),tf.float32)
    im4 =  tf.cast(tf.image.resize_images(im4,(299,299)),tf.float32)
    im5 =  tf.cast(tf.image.resize_images(im5,(299,299)),tf.float32)
    
    im6 = tf.image.flip_left_right(im1)
    im7 = tf.image.flip_left_right(im2)
    im8 = tf.image.flip_left_right(im3)
    im9 = tf.image.flip_left_right(im4)
    im10 = tf.image.flip_left_right(im5)
    
    return tf.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9,im10])

ims = tf.map_fn(fn=read_images,elems=tf_filepath2,dtype=np.float32)
ims = tf.reshape(ims,(batch,)+input_size)/255.0
sample_im = ims * 1

def data_in_train():
    return tf.map_fn(fn = train_preproc,elems = ims,dtype=np.float32)      

def data_in_test():
    return tf.map_fn(fn = test_preproc,elems = ims,dtype=np.float32)

data_in = tf.cond(
        is_training,
        true_fn = data_in_train,
        false_fn = data_in_test
        )
    

test_batches = [filepaths[i:i + batch] for i in range(0, len(filepaths), batch)]
ground_truth_class_batches = [f_class[i:i + batch] for i in range(0, len(f_class), batch)]
ground_truth_order_batches = [f_order[i:i + batch] for i in range(0, len(f_order), batch)]
ground_truth_family_batches = [f_family[i:i + batch] for i in range(0, len(f_family), batch)]
ground_truth_genus_batches = [f_genus[i:i + batch] for i in range(0, len(f_genus), batch)]
ground_truth_species_batches = [f_species[i:i + batch] for i in range(0, len(f_species), batch)]


# =============================================================================
# Construct network 
# =============================================================================
with slim.arg_scope(inception_resnet_v2_arg_scope()):    
    logits,endpoints = inception_resnet_v2(data_in,
                                        num_classes=numclasses,
                                        is_training=is_training)


    embs_bn = tf.layers.batch_normalization(endpoints['PreLogitsFlatten'], training=is_training2)
    
    feat_500 = slim.fully_connected(
                    inputs=embs_bn,
                    num_outputs=500,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    weights_regularizer=None, 
                    trainable=True,
                    scope='feat_500'                            
            )

    logits_species = slim.fully_connected(feat_500,numclasses,activation_fn=None,weights_regularizer=None,
                                scope='species')
   
    logits_family = slim.fully_connected(feat_500,357,activation_fn=None,weights_regularizer=None,
                                scope='family')
    logits_genus = slim.fully_connected(feat_500,3210,activation_fn=None,weights_regularizer=None,
                                scope='genus')
    logits_class = slim.fully_connected(feat_500,8,activation_fn=None,weights_regularizer=None,
                                scope='class')
    logits_order = slim.fully_connected(feat_500,78,activation_fn=None,weights_regularizer=None,
                                scope='order') 


feat_class = tf.nn.softmax(logits_class)
feat_order = tf.nn.softmax(logits_order)
feat_family = tf.nn.softmax(logits_family)
feat_genus = tf.nn.softmax(logits_genus)
feat_species = tf.nn.softmax(logits_species)

    
variables_to_restore = slim.get_variables_to_restore()
restorer = tf.train.Saver(variables_to_restore)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
print(f"[{datetimestr()}] Start process")

# ----- Write csv ----- #
with open(prediction_family_csv, 'a', newline='', encoding="utf-8") as csv1:
    writer = csv.writer(csv1, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ['filepath', 'groundtruth', 'top1 family', 'top1 probability', "top5 family", "top5 probability", "top1 score", "top5 score"]
    writer.writerow(header) 
    
    with open(prediction_genus_csv, 'a', newline='', encoding="utf-8") as csv2:
        writer2 = csv.writer(csv2, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header2 = ['filepath', 'groundtruth', 'top1 genus', 'top1 probability', "top5 genus", "top5 probability", "top1 score", "top5 score"]
        writer2.writerow(header2) 

        with open(prediction_species_csv, 'a', newline='', encoding="utf-8") as csv3:
            writer3 = csv.writer(csv3, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header3 = ['filepath', 'groundtruth', 'top1 species', 'top1 probability', "top5 species", "top5 probability", "top1 score", "top5 score"]
            writer3.writerow(header3)   
            
            with open(prediction_class_csv, 'a', newline='', encoding="utf-8") as csv4:
                writer4 = csv.writer(csv4, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                header4 = ['filepath', 'groundtruth', 'top1 class', 'top1 probability', "top5 class", "top5 probability", "top1 score", "top5 score"]
                writer4.writerow(header4)   
                
                with open(prediction_order_csv, 'a', newline='', encoding="utf-8") as csv5:
                    writer5 = csv.writer(csv5, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    header5 = ['filepath', 'groundtruth', 'top1 order', 'top1 probability', "top5 order", "top5 probability", "top1 score", "top5 score"]
                    writer5.writerow(header5)  
    
                    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                        
                        sess.run(tf.global_variables_initializer())
                        restorer.restore(sess, model_checkpoint) 
                        
                        batch_counter = 0
                        
                        top1_class_counter = 0
                        top5_class_counter = 0
                        
                        top1_order_counter = 0
                        top5_order_counter = 0
                        
                        top1_family_counter = 0
                        top5_family_counter = 0
                        
                        top1_genus_counter = 0
                        top5_genus_counter = 0  
                        
                        top1_species_counter = 0
                        top5_species_counter = 0

                        filepath_list = []
                        
                        for current_batch_files, ground_truths_class, ground_truths_order, ground_truths_family, ground_truths_genus, ground_truths_species in zip(test_batches, ground_truth_class_batches, ground_truth_order_batches, ground_truth_family_batches, ground_truth_genus_batches, ground_truth_species_batches):
                            
                            iter_run = len(current_batch_files)//global_batch
                            
                            print(f"[{datetimestr()}] Files:{len(current_batch_files)} Batch counter:{batch_counter}")
                    
                            if len(current_batch_files) > (iter_run * global_batch):            
                                iter_run += 1
                                padded = (iter_run * global_batch) - len(current_batch_files) 
                                current_batch_files = current_batch_files + ([current_batch_files[0]] * padded)
                                ground_truths_class = ground_truths_class + ([ground_truths_class[0]] * padded)
                                ground_truths_order = ground_truths_order + ([ground_truths_order[0]] * padded)
                                ground_truths_family = ground_truths_family + ([ground_truths_family[0]] * padded)
                                ground_truths_genus = ground_truths_genus + ([ground_truths_genus[0]] * padded)
                                ground_truths_species = ground_truths_species + ([ground_truths_species[0]] * padded)
                            else:
                                padded = 0
                            
                            
                            c = 0
                            for n in range(iter_run):

                                paths = current_batch_files[n*global_batch:(n*global_batch)+global_batch]
                                gtruths_class = ground_truths_class[n*global_batch:(n*global_batch)+global_batch] 
                                gtruths_order = ground_truths_order[n*global_batch:(n*global_batch)+global_batch] 
                                gtruths_family = ground_truths_family[n*global_batch:(n*global_batch)+global_batch]            
                                gtruths_genus = ground_truths_genus[n*global_batch:(n*global_batch)+global_batch]
                                gtruths_species = ground_truths_species[n*global_batch:(n*global_batch)+global_batch]  
                                
                        
                        
                                pred_class, pred_order, pred_family, pred_genus, pred_species = sess.run(
                                            [feat_class, feat_order, feat_family, feat_genus, feat_species],
                                            feed_dict = {
                                                        tf_filepath2:paths,   
                                                        is_training : False,
                                                        is_training2 : False
                                                    }
                                        )

                                pred_class_gbatch = np.reshape(pred_class,(global_batch, 10, -1))
                                pred_class_gbatch_mean = np.mean(pred_class_gbatch,axis=1)
                                
                                pred_order_gbatch = np.reshape(pred_order,(global_batch, 10, -1))
                                pred_order_gbatch_mean = np.mean(pred_order_gbatch,axis=1)                                
                                
                                pred_family_gbatch = np.reshape(pred_family,(global_batch, 10, -1))
                                pred_family_gbatch_mean = np.mean(pred_family_gbatch,axis=1)
                                
                                pred_genus_gbatch = np.reshape(pred_genus,(global_batch, 10, -1))
                                pred_genus_gbatch_mean = np.mean(pred_genus_gbatch,axis=1)
                                
                                pred_species_gbatch = np.reshape(pred_species,(global_batch, 10, -1))
                                pred_species_gbatch_mean = np.mean(pred_species_gbatch,axis=1)
                                
                                if n == (iter_run - 1): 
                                    score1 = 0
                                    score5 = 0
                                    # =============================================================================
                                    #                 family
                                    # =============================================================================
                                    for i,a in enumerate(pred_family_gbatch_mean[0:(global_batch-padded)]):
 
                                        filepath_list.append(paths[i])                    
                                        c += 1
                                        
                                        current_family = int(gtruths_family[i])
                                        topN_family = a.argsort()[-topN:][::-1]
                                        topN_family_probability = np.sort(a)[-topN:][::-1] 
                                        
                                        if current_family in topN_family:
                                            top5_family_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_family == topN_family[0]:
                                            top1_family_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                        
                                        writer.writerow([paths[i], current_family, topN_family[0], topN_family_probability[0], topN_family, topN_family_probability, score1, score5])
                    
                                    # =============================================================================
                                    #                 genus
                                    # =============================================================================
                                    for i,a in enumerate(pred_genus_gbatch_mean[0:(global_batch-padded)]):
                    
                                        current_genus = int(gtruths_genus[i])
                                        topN_genus = a.argsort()[-topN:][::-1]
                                        topN_genus_probability = np.sort(a)[-topN:][::-1] 
                                        
                                        if current_genus in topN_genus:
                                            top5_genus_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_genus == topN_genus[0]:
                                            top1_genus_counter += 1 
                                            score1 = 1
                                        else:
                                            score1 = 0
                                            
                                        writer2.writerow([paths[i], current_genus, topN_genus[0], topN_genus_probability[0], topN_genus, topN_genus_probability, score1, score5])
                    
                                    # =============================================================================
                                    #                 species
                                    # =============================================================================
                                    for i,a in enumerate(pred_species_gbatch_mean[0:(global_batch-padded)]):
                                        current_species = int(gtruths_species[i])
                                        
                                        topN_species = a.argsort()[-topN:][::-1]
                                        topN_species_probability = np.sort(a)[-topN:][::-1]
                                        
                                        
                                        if current_species in topN_species:
                                            top5_species_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_species == topN_species[0]:
                                            top1_species_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                        
                                        writer3.writerow([paths[i], current_species, topN_species[0], topN_species_probability[0], topN_species, topN_species_probability, score1, score5])
                                     
                                    # =============================================================================
                                    #                 class
                                    # =============================================================================
                                    for i,a in enumerate(pred_class_gbatch_mean[0:(global_batch-padded)]):
                                        current_class = int(gtruths_class[i])
                                        
                                        topN_class = a.argsort()[-topN:][::-1]
                                        topN_class_probability = np.sort(a)[-topN:][::-1]
                                        
                                        
                                        if current_class in topN_class:
                                            top5_class_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_class == topN_class[0]:
                                            top1_class_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                        
                                        writer4.writerow([paths[i], current_class, topN_class[0], topN_class_probability[0], topN_class, topN_class_probability, score1, score5])                                        


                                    # =============================================================================
                                    #                 order
                                    # =============================================================================
                                    for i,a in enumerate(pred_order_gbatch_mean[0:(global_batch-padded)]):
                                        current_order = int(gtruths_order[i])
                                        
                                        topN_order = a.argsort()[-topN:][::-1]
                                        topN_order_probability = np.sort(a)[-topN:][::-1]
                                        
                                        
                                        if current_order in topN_order:
                                            top5_order_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_order == topN_order[0]:
                                            top1_order_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                        
                                        writer5.writerow([paths[i], current_order, topN_order[0], topN_order_probability[0], topN_order, topN_order_probability, score1, score5])                                        
                                        
                                else:  
                                    # =============================================================================
                                    #                 family
                                    # =============================================================================
                                    for i,a in enumerate(pred_family_gbatch_mean):
                                        filepath_list.append(paths[i])
                                        c += 1
                    
                                        current_family = int(gtruths_family[i])
                                        topN_family = a.argsort()[-topN:][::-1]
                                        topN_family_probability = np.sort(a)[-topN:][::-1] 
                                        
                                        if current_family in topN_family:
                                            top5_family_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_family == topN_family[0]:
                                            top1_family_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                            
                                        writer.writerow([paths[i], current_family, topN_family[0], topN_family_probability[0], topN_family, topN_family_probability, score1, score5])                            
                                    
                                    # =============================================================================
                                    #                 genus
                                    # =============================================================================
                                    for i,a in enumerate(pred_genus_gbatch_mean):
                    
                                        current_genus = int(gtruths_genus[i])
                                        topN_genus = a.argsort()[-topN:][::-1]
                                        topN_genus_probability = np.sort(a)[-topN:][::-1] 
                                        
                                        if current_genus in topN_genus:
                                            top5_genus_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_genus == topN_genus[0]:
                                            top1_genus_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                        
                                        writer2.writerow([paths[i], current_genus, topN_genus[0], topN_genus_probability[0], topN_genus, topN_genus_probability, score1, score5])                            
                                    
                                    # =============================================================================
                                    #                 species
                                    # =============================================================================
                                    for i,a in enumerate(pred_species_gbatch_mean):
                                        current_species = int(gtruths_species[i])
                                        
                                        topN_species = a.argsort()[-topN:][::-1]
                                        topN_species_probability = np.sort(a)[-topN:][::-1]
                                        
                    
                                        if current_species in topN_species:
                                            top5_species_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_species == topN_species[0]:
                                            top1_species_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                            
                                        writer3.writerow([paths[i], current_species, topN_species[0], topN_species_probability[0], topN_species, topN_species_probability, score1, score5])
                                        

                                    # =============================================================================
                                    #                 class
                                    # =============================================================================
                                    for i,a in enumerate(pred_class_gbatch_mean):
                                        current_class = int(gtruths_class[i])
                                        
                                        topN_class = a.argsort()[-topN:][::-1]
                                        topN_class_probability = np.sort(a)[-topN:][::-1]
                                        
                                        
                                        if current_class in topN_class:
                                            top5_class_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_class == topN_class[0]:
                                            top1_class_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                        
                                        writer4.writerow([paths[i], current_class, topN_class[0], topN_class_probability[0], topN_class, topN_class_probability, score1, score5])                                        


                                    # =============================================================================
                                    #                 order
                                    # =============================================================================
                                    for i,a in enumerate(pred_order_gbatch_mean):
                                        current_order = int(gtruths_order[i])

                                        
                                        topN_order = a.argsort()[-topN:][::-1]
                                        topN_order_probability = np.sort(a)[-topN:][::-1]
                                        
                                        
                                        if current_order in topN_order:
                                            top5_order_counter += 1
                                            score5 = 1
                                        else:
                                            score5 = 0
                                        if current_order == topN_order[0]:
                                            top1_order_counter += 1
                                            score1 = 1
                                        else:
                                            score1 = 0
                                        
                                        writer5.writerow([paths[i], current_order, topN_order[0], topN_order_probability[0], topN_order, topN_order_probability, score1, score5])                                        
                            
                            
                            print("Top-1 class:", round(top1_class_counter / len(filepath_list),4), top1_class_counter,"/",len(filepath_list), " ----- Top-5 class:", round(top5_class_counter / len(filepath_list),4), top5_class_counter,"/",len(filepath_list))
                            print("Top-1 order:", round(top1_order_counter / len(filepath_list),4), top1_order_counter,"/",len(filepath_list), " ----- Top-5 order:", round(top5_order_counter / len(filepath_list),4), top5_order_counter,"/",len(filepath_list))
                            print("Top-1 family:", round(top1_family_counter / len(filepath_list),4), top1_family_counter,"/",len(filepath_list), " ----- Top-5 family:", round(top5_family_counter / len(filepath_list),4), top5_family_counter,"/",len(filepath_list))
                            print("Top-1 genus:", round(top1_genus_counter / len(filepath_list),4), top1_genus_counter,"/",len(filepath_list), " ----- Top-5 genus:", round(top5_genus_counter / len(filepath_list),4), top5_genus_counter,"/",len(filepath_list))
                            print("Top-1 species:", round(top1_species_counter / len(filepath_list),4), top1_species_counter,"/",len(filepath_list), " ----- Top-5 species:", round(top5_species_counter / len(filepath_list),4), top5_species_counter,"/",len(filepath_list))
                            
                            batch_counter +=1
                            




