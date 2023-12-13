# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:42:05 2023

@author: user
"""

import os 
import sys
sys.path.append("path/to/tf_slim/models/research/slim")
import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim

import numpy as np
from nets.inception_resnet_v2 import inception_resnet_v2
from nets.inception_resnet_v2 import inception_resnet_v2_arg_scope
from database_module import database_module




class inception_resnetv2_module(object):
    def __init__(self,
                 batch,
                 iterbatch,
                 numclass,
                 numorder,
                 numfamily,
                 numgenus,
                 numclasses,
                 image_dir_parent_train,
                 image_dir_parent_test,
                 train_file,
                 test_file,
                 input_size,
                 checkpoint_model,
                 learning_rate,
                 save_dir,
                 max_iter,
                 val_freq,
                 val_iter
                 ):
        
        self.batch = batch
        self.iterbatch = iterbatch
        self.image_dir_parent_train = image_dir_parent_train
        self.image_dir_parent_test = image_dir_parent_test
        self.train_file = train_file
        self.test_file = test_file
        self.input_size = input_size
        self.numclass = numclass
        self.numorder = numorder
        self.numfamily = numfamily
        self.numgenus = numgenus
        self.numclasses = numclasses
        self.checkpoint_model = checkpoint_model
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.max_iter = max_iter
        self.val_freq = val_freq
        self.val_iter = val_iter

        

        print('Initiating database...')
        
        self.train_database = database_module(
                image_source_dir = self.image_dir_parent_train,
                database_file = self.train_file,
                batch = self.batch,
                input_size = self.input_size,
                shuffle = True)

        self.test_database = database_module(
                image_source_dir = self.image_dir_parent_test,
                database_file = self.test_file,
                batch = self.batch,
                input_size = self.input_size,
                shuffle = True)
        
         
        print('Initiating tensors...')
        x = tf.placeholder(tf.float32,(self.batch,) + self.input_size)
        y1 = tf.placeholder(tf.int32,(self.batch,)) # species
        y2 = tf.placeholder(tf.int32,(self.batch,)) # family
        y3 = tf.placeholder(tf.int32,(self.batch,)) # genus
        y4 = tf.placeholder(tf.int32,(self.batch,)) # class
        y5 = tf.placeholder(tf.int32,(self.batch,)) # order
        y_onehot1 = tf.one_hot(y1,self.numclasses)
        y_onehot2 = tf.one_hot(y2,self.numfamily)
        y_onehot3 = tf.one_hot(y3,self.numgenus)
        y_onehot4 = tf.one_hot(y4,self.numclass)
        y_onehot5 = tf.one_hot(y5,self.numorder)
        self.is_training = tf.placeholder(tf.bool)
        self.is_training2 = tf.placeholder(tf.bool)


        train_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=True)
        
        def data_in_train():
            return tf.map_fn(fn = train_preproc,elems = x,dtype=np.float32)
        
        test_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=False)        
        
        def data_in_test():
            return tf.map_fn(fn = test_preproc,elems = x,dtype=np.float32)
        
        data_in = tf.cond(
                self.is_training,
                true_fn = data_in_train,
                false_fn = data_in_test
                )

        print('Constructing network...')        

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits,endpoints = inception_resnet_v2(data_in,
                                            num_classes=self.numclasses,
                                            is_training=self.is_training)
            
            embs_bn = tf.layers.batch_normalization(endpoints['PreLogitsFlatten'], training=self.is_training2)
            
            feat_500 = slim.fully_connected(
                            inputs=embs_bn,
                            num_outputs=500,
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.00004), 
                            trainable=True,
                            scope='feat_500'                            
                    )

            logits_species = slim.fully_connected(feat_500,self.numclasses,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.00004),
                                        scope='species')
           
            logits_family = slim.fully_connected(feat_500,self.numfamily,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.00004),
                                        scope='family')
            logits_genus = slim.fully_connected(feat_500,self.numgenus,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.00004),
                                        scope='genus')
            logits_class = slim.fully_connected(feat_500,self.numclass,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.00004),
                                        scope='class')
            logits_order = slim.fully_connected(feat_500,self.numorder,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.00004),
                                        scope='order')            


        self.var_list = [v for v in tf.trainable_variables()]



        # ----- Get all variables ----- #
        self.variables_trainable = tf.trainable_variables()

        self.var_list_front = self.variables_trainable[:-22]
        self.var_list_end = self.variables_trainable[-22:]
        
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
   
        self.var_list_train = self.var_list_front + self.var_list_end


        
            
        with tf.name_scope("cross_entropy"): 
            with tf.name_scope("auxloss"):
                self.auxloss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=endpoints['AuxLogits'], labels=y_onehot1))
            with tf.name_scope("logits_loss_species"):
                self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_species, labels=y_onehot1))

            with tf.name_scope("logits_loss_family"):
                self.loss_family = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_family, labels=y_onehot2))

            with tf.name_scope("logits_loss_genus"):
                self.loss_genus = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_genus, labels=y_onehot3))

            with tf.name_scope("logits_loss_class"):
                self.loss_class = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_class, labels=y_onehot4))
                                
            with tf.name_scope("logits_loss_order"):
                self.loss_order = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_order, labels=y_onehot5))                                

            with tf.name_scope("L2_reg_loss"):
                 self.regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
                
                
            with tf.name_scope("total_loss"):
                self.totalloss = self.loss + self.loss_family + self.loss_genus + self.loss_class + self.loss_order + self.auxloss + self.regularization_loss

            

        with tf.name_scope("accuracy"):
            with tf.name_scope('accuracy_species'):
                prediction = tf.argmax(logits_species,1)
                match = tf.equal(prediction,tf.argmax(y_onehot1,1))
                self.accuracy = tf.reduce_mean(tf.cast(match,tf.float32))  
            with tf.name_scope('accuracy_family'):
                prediction2 = tf.argmax(logits_family,1)
                match = tf.equal(prediction2,tf.argmax(y_onehot2,1))            
                self.accuracy_family = tf.reduce_mean(tf.cast(match,tf.float32)) 
            with tf.name_scope('accuracy_genus'):       
                prediction3 = tf.argmax(logits_genus,1)
                match = tf.equal(prediction3,tf.argmax(y_onehot3,1))            
                self.accuracy_genus = tf.reduce_mean(tf.cast(match,tf.float32))    
            with tf.name_scope('accuracy_class'):       
                prediction4 = tf.argmax(logits_class,1)
                match = tf.equal(prediction4,tf.argmax(y_onehot4,1))            
                self.accuracy_class = tf.reduce_mean(tf.cast(match,tf.float32))
            with tf.name_scope('accuracy_order'):       
                prediction5 = tf.argmax(logits_order,1)
                match = tf.equal(prediction5,tf.argmax(y_onehot5,1))            
                self.accuracy_order = tf.reduce_mean(tf.cast(match,tf.float32))                   
            


        

        
        
        # only load the conv layer but not the logits
        self.variables_to_restore = slim.get_variables_to_restore()
        restore_fn = slim.assign_from_checkpoint_fn(
            self.checkpoint_model, self.variables_to_restore[:-28])
      
   
        
        with tf.name_scope("train"):

            loss_accumulator = tf.Variable(0.0, trainable=False)
            loss_accumulator_class = tf.Variable(0.0, trainable=False)
            loss_accumulator_order = tf.Variable(0.0, trainable=False)
            loss_accumulator_family = tf.Variable(0.0, trainable=False)
            loss_accumulator_genus = tf.Variable(0.0, trainable=False)
            loss_accumulator_species = tf.Variable(0.0, trainable=False)
            loss_accumulator_aux = tf.Variable(0.0, trainable=False)
            loss_accumulator_reg = tf.Variable(0.0, trainable=False)
            
            acc_accumulator = tf.Variable(0.0, trainable=False)
            acc_accumulator_family = tf.Variable(0.0, trainable=False)
            acc_accumulator_genus = tf.Variable(0.0, trainable=False)
            acc_accumulator_class = tf.Variable(0.0, trainable=False)
            acc_accumulator_order = tf.Variable(0.0, trainable=False)
            
            self.collect_loss = loss_accumulator.assign_add(self.totalloss)
            
            self.collect_loss_class = loss_accumulator_class.assign_add(self.loss_class)
            self.collect_loss_order = loss_accumulator_order.assign_add(self.loss_order)
            self.collect_loss_family = loss_accumulator_family.assign_add(self.loss_family)
            self.collect_loss_genus = loss_accumulator_genus.assign_add(self.loss_genus)
            self.collect_loss_species = loss_accumulator_species.assign_add(self.loss)
            self.collect_loss_aux = loss_accumulator_aux.assign_add(self.auxloss)
            self.collect_loss_reg = loss_accumulator_reg.assign_add(self.regularization_loss)
            
            
            self.collect_acc = acc_accumulator.assign_add(self.accuracy)
            self.collect_acc_family = acc_accumulator_family.assign_add(self.accuracy_family)
            self.collect_acc_genus = acc_accumulator_genus.assign_add(self.accuracy_genus)
            self.collect_acc_class = acc_accumulator_class.assign_add(self.accuracy_class)
            self.collect_acc_order = acc_accumulator_order.assign_add(self.accuracy_order)

                        
            self.average_loss = tf.cond(self.is_training,
                                        lambda: loss_accumulator / self.iterbatch,
                                        lambda: loss_accumulator / self.val_iter)

            self.average_loss_class = tf.cond(self.is_training,
                                        lambda: loss_accumulator_class / self.iterbatch,
                                        lambda: loss_accumulator_class / self.val_iter)

            self.average_loss_order = tf.cond(self.is_training,
                                        lambda: loss_accumulator_order / self.iterbatch,
                                        lambda: loss_accumulator_order / self.val_iter)
            
            self.average_loss_family = tf.cond(self.is_training,
                                        lambda: loss_accumulator_family / self.iterbatch,
                                        lambda: loss_accumulator_family / self.val_iter)

            self.average_loss_genus = tf.cond(self.is_training,
                                        lambda: loss_accumulator_genus / self.iterbatch,
                                        lambda: loss_accumulator_genus / self.val_iter)  
            
            self.average_loss_species = tf.cond(self.is_training,
                                        lambda: loss_accumulator_species / self.iterbatch,
                                        lambda: loss_accumulator_species / self.val_iter)            

            self.average_loss_aux = tf.cond(self.is_training,
                                        lambda: loss_accumulator_aux / self.iterbatch,
                                        lambda: loss_accumulator_aux / self.val_iter)  

            self.average_loss_reg = tf.cond(self.is_training,
                                        lambda: loss_accumulator_reg / self.iterbatch,
                                        lambda: loss_accumulator_reg / self.val_iter)              
            
            self.average_acc = tf.cond(self.is_training,
                                       lambda: acc_accumulator / self.iterbatch,
                                       lambda: acc_accumulator / self.val_iter)
            self.average_acc_family = tf.cond(self.is_training,
                                       lambda: acc_accumulator_family / self.iterbatch,
                                       lambda: acc_accumulator_family / self.val_iter)
            self.average_acc_genus = tf.cond(self.is_training,
                                       lambda: acc_accumulator_genus / self.iterbatch,
                                       lambda: acc_accumulator_genus / self.val_iter)
            self.average_acc_class = tf.cond(self.is_training,
                                       lambda: acc_accumulator_class / self.iterbatch,
                                       lambda: acc_accumulator_class / self.val_iter)
            self.average_acc_order = tf.cond(self.is_training,
                                       lambda: acc_accumulator_order / self.iterbatch,
                                       lambda: acc_accumulator_order / self.val_iter)            

            
            self.zero_op_loss = tf.assign(loss_accumulator,0.0)
            self.zero_op_loss_class = tf.assign(loss_accumulator_class,0.0)
            self.zero_op_loss_order = tf.assign(loss_accumulator_order,0.0)
            self.zero_op_loss_family = tf.assign(loss_accumulator_family,0.0)
            self.zero_op_loss_genus = tf.assign(loss_accumulator_genus,0.0)
            self.zero_op_loss_species = tf.assign(loss_accumulator_species,0.0)
            self.zero_op_loss_aux = tf.assign(loss_accumulator_aux,0.0)
            self.zero_op_loss_reg = tf.assign(loss_accumulator_reg,0.0)
            
            self.zero_op_acc = tf.assign(acc_accumulator,0.0)
            self.zero_op_acc_family = tf.assign(acc_accumulator_family,0.0)
            self.zero_op_acc_genus = tf.assign(acc_accumulator_genus,0.0)
            self.zero_op_acc_class = tf.assign(acc_accumulator_class,0.0)
            self.zero_op_acc_order = tf.assign(acc_accumulator_order,0.0)
            
            
            # ----- Separate vars ----- #      
            self.accum_train_front = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_front] 
            self.accum_train_end = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_end]                                               
        
            self.zero_ops_train_front = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_front]
            self.zero_ops_train_end = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_end]

            
            with tf.control_dependencies(self.update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate * 0.1)
                
                # create another optimizer
                optimizer_end_layers = tf.train.AdamOptimizer(self.learning_rate)

                # compute gradient with an other list of var_list
                gradient1 = optimizer.compute_gradients(self.totalloss,self.var_list_front)
                gradient2 = optimizer_end_layers.compute_gradients(self.totalloss,self.var_list_end)
                
                gradient_only_front = [gc[0] for gc in gradient1]
                gradient_only_front,_ = tf.clip_by_global_norm(gradient_only_front,1.25)
                
                gradient_only_back = [gc[0] for gc in gradient2]
                gradient_only_back,_ = tf.clip_by_global_norm(gradient_only_back,1.25)
                

               
                self.accum_train_ops_front = [self.accum_train_front[i].assign_add(gc) for i,gc in enumerate(gradient_only_front)]
            
                self.accum_train_ops_end = [self.accum_train_end[i].assign_add(gc) for i,gc in enumerate(gradient_only_back)]
                
                
                
                
            # ----- Apply gradients ----- #
            self.train_step_front = optimizer.apply_gradients(
                    [(self.accum_train_front[i], gc[1]) for i, gc in enumerate(gradient1)])
      
            self.train_step_end = optimizer_end_layers.apply_gradients(
                    [(self.accum_train_end[i], gc[1]) for i, gc in enumerate(gradient2)])

            
            

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        
        # ----- Create saver ----- #
        saver = tf.train.Saver(var_list=var_list, max_to_keep=0)

        tf.summary.scalar('loss',self.average_loss) 
        tf.summary.scalar('accuracy_species',self.average_acc) 
        tf.summary.scalar('accuracy_family',self.average_acc_family)
        tf.summary.scalar('accuracy_genus',self.average_acc_genus)
        tf.summary.scalar('accuracy_class',self.average_acc_class)
        tf.summary.scalar('accuracy_order',self.average_acc_order)

        self.merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'accuracy'),
                                        tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
        tensorboar_dir = os.path.join(self.save_dir,'%s_tensorboard'%current_time())
        writer_train = tf.summary.FileWriter(tensorboar_dir+'/train')
        writer_test = tf.summary.FileWriter(tensorboar_dir+'/test')


        print('Commencing training...') 
        val_best = 0.0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer_train.add_graph(sess.graph)
            writer_test.add_graph(sess.graph)
            
            restore_fn(sess)
            
            for i in range(self.max_iter+1):
                try:
                    sess.run(self.zero_ops_train_front)
                    sess.run(self.zero_ops_train_end)
                    sess.run([self.zero_op_acc,self.zero_op_acc_family,self.zero_op_acc_genus,self.zero_op_acc_class,self.zero_op_acc_order,self.zero_op_loss,
                              self.zero_op_loss_class,self.zero_op_loss_order,self.zero_op_loss_family,self.zero_op_loss_genus,self.zero_op_loss_species,self.zero_op_loss_aux,self.zero_op_loss_reg])
                    
                    # validations
                    if i % self.val_freq == 0:                        
                        print('Start:%f'%sess.run(loss_accumulator))
                        for j in range(self.val_iter):
                            img,lbl1,lbl2,lbl3,lbl4,lbl5 = self.test_database.read_batch()
                            sess.run(
                                        [self.collect_loss,self.collect_loss_class,self.collect_loss_order,self.collect_loss_family,self.collect_loss_genus,self.collect_loss_species,self.collect_loss_aux,self.collect_loss_reg,
                                         self.collect_acc,self.collect_acc_family,self.collect_acc_genus,self.collect_acc_class, self.collect_acc_order],
                                        feed_dict = {x : img,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     y3 : lbl3,
                                                     y4 : lbl4,
                                                     y5 : lbl5,
                                                     self.is_training : False,
                                                     self.is_training2 : False
                                        }                                  
                                    )
                            print('[%i]:%f'%(j,sess.run(loss_accumulator)))
                        print('End:%f'%sess.run(loss_accumulator))  
                        s,self.netLoss,self.netAccuracy,self.netAccuracyFamily,self.netAccuracyGenus,self.netAccuracyClass,self.netAccuracyOrder,self.netLossClass,self.netLossOrder,self.netLossFamily,self.netLossGenus,self.netLossSpecies,self.netLossAux,self.netLossReg = sess.run(
                            [self.merged,self.average_loss,self.average_acc,self.average_acc_family,self.average_acc_genus,self.average_acc_class,self.average_acc_order,
                             self.average_loss_class,self.average_loss_order,self.average_loss_family,self.average_loss_genus,self.average_loss_species,self.average_loss_aux,self.average_loss_reg],
                                    feed_dict = {
                                            self.is_training : False                                            
                                    }                            
                                ) 
                        writer_test.add_summary(s, i)
                        print('[Valid] Total loss: %f'%(self.netLoss))
                        print('[Valid] Auxillary loss: %f'%(self.netLossAux))
                        print('[Valid] Regularization loss: %f'%(self.netLossReg))
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Species:%f'%(self.test_database.epoch,i,self.netLossSpecies,self.netAccuracy)) 
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Family:%f'%(self.test_database.epoch,i,self.netLossFamily,self.netAccuracyFamily)) 
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Genus:%f'%(self.test_database.epoch,i,self.netLossGenus,self.netAccuracyGenus)) 
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Class:%f'%(self.test_database.epoch,i,self.netLossClass,self.netAccuracyClass)) 
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Order:%f'%(self.test_database.epoch,i,self.netLossOrder,self.netAccuracyOrder))



                        sess.run([self.zero_op_acc,self.zero_op_acc_family,self.zero_op_acc_genus,self.zero_op_acc_class,self.zero_op_acc_order,self.zero_op_loss,
                                  self.zero_op_loss_class,self.zero_op_loss_order,self.zero_op_loss_family,self.zero_op_loss_genus,self.zero_op_loss_species,self.zero_op_loss_aux,self.zero_op_loss_reg])
                        
                        if self.netAccuracy > val_best:
                            val_best = self.netAccuracy
                            saver.save(sess, os.path.join(self.save_dir,'best.ckpt'))
                            print('Best model saved')





                    # training
                    for j in range(self.iterbatch):
                        img,lbl1,lbl2,lbl3,lbl4,lbl5 = self.train_database.read_batch()
                        

    
                        sess.run(
                                    [self.collect_loss,self.collect_loss_class,self.collect_loss_order,self.collect_loss_family,self.collect_loss_genus,self.collect_loss_species,self.collect_loss_aux,self.collect_loss_reg,
                                     self.collect_acc,self.collect_acc_family,self.collect_acc_genus,self.collect_acc_class,self.collect_acc_order,self.accum_train_ops_front,self.accum_train_ops_end],
                                    feed_dict = {x : img,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     y3 : lbl3,
                                                     y4 : lbl4,
                                                     y5 : lbl5,
                                                 self.is_training : True,
                                                 self.is_training2 : True,
                                    }                                
                                )
                        
                    s,self.netLoss,self.netAccuracy,self.netAccuracyFamily,self.netAccuracyGenus,self.netAccuracyClass,self.netAccuracyOrder,self.netLossClass,self.netLossOrder,self.netLossFamily,self.netLossGenus,self.netLossSpecies,self.netLossAux,self.netLossReg = sess.run(
                            [self.merged,self.average_loss,self.average_acc,self.average_acc_family,self.average_acc_genus,self.average_acc_class,self.average_acc_order,
                             self.average_loss_class,self.average_loss_order,self.average_loss_family,self.average_loss_genus,self.average_loss_species,self.average_loss_aux,self.average_loss_reg],
                                feed_dict = {
                                        self.is_training : True                                    
                                }                            
                            ) 
                    writer_train.add_summary(s, i)
                    
                    
                    sess.run([self.train_step_front])
                    sess.run([self.train_step_end])
                        
                    print('[Train] Total loss: %f'%(self.netLoss))
                    print('[Train] Auxillary loss: %f'%(self.netLossAux))
                    print('[Train] Regularization loss: %f'%(self.netLossReg))
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Species:%f'%(self.train_database.epoch,i,self.netLossSpecies,self.netAccuracy))
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Family:%f'%(self.train_database.epoch,i,self.netLossFamily,self.netAccuracyFamily)) 
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Genus:%f'%(self.train_database.epoch,i,self.netLossGenus,self.netAccuracyGenus)) 
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Class:%f'%(self.train_database.epoch,i,self.netLossClass,self.netAccuracyClass)) 
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Order:%f'%(self.train_database.epoch,i,self.netLossOrder,self.netAccuracyOrder)) 

                    
                    if i % 50000 == 0:
                        saver.save(sess, os.path.join(self.save_dir,'%06i.ckpt'%i)) 
                    
                except KeyboardInterrupt:
                    print('Interrupt detected. Ending...')
                    break
                
            saver.save(sess, os.path.join(self.save_dir,'final.ckpt')) 
            print('Model saved')


        


        



