from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import os
import math
import scipy.misc as sic
from PIL import Image
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib as contrib
    
def generator(FLAGS, target, init=None, content=None,reuse=False, training=True):
    if init is not None:
        if FLAGS.task_mode == 'texture_synthesis':
            var = tf.get_variable('gen_img', initializer = init)   
        elif FLAGS.task_mode == 'style_transfer':
            var = tf.get_variable('gen_img', initializer = init + tf.random_normal(tf.shape(init), 0, FLAGS.stddev))
    else:
        if FLAGS.task_mode == 'texture_synthesis':
            if FLAGS.texture_shape == [-1,-1]:
                shape = [1,target.shape[1],target.shape[2],3]
            else:
                shape = [1,FLAGS.texture_shape[0],FLAGS.texture_shape[1],3]
            var = tf.get_variable('gen_img',shape=shape, \
                     #initializer = tf.truncated_normal_initializer(0,0.2),\
                      initializer = tf.random_normal_initializer(0,0.5),
                                   dtype=tf.float32,trainable=True, collections=None)   
        elif FLAGS.task_mode == 'style_transfer':
            print("test point!")
            shape = [1,content.shape[1],content.shape[2],3]
            var = tf.get_variable('gen_img',shape=shape, \
                     initializer = tf.truncated_normal_initializer(0,0.2),\
                                   dtype=tf.float32,trainable=True, collections=None) 
    return tf.tanh(var)
    
# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        image_raw = Image.open(FLAGS.target_dir)
        if image_raw.mode is not 'RGB':
            image_raw = image_raw.convert('RGB')
        image_raw = np.asarray(image_raw)/255
        targets = tf.constant(image_raw)  
        targets = tf.image.convert_image_dtype(targets, dtype = tf.float32, saturate = True)
        targets = preprocess(targets)  
        samples = tf.expand_dims(targets, axis=0) 
        if FLAGS.content_dir is not None:
            image_raw = np.asarray(Image.open(FLAGS.content_dir))/255
            contents = tf.constant(image_raw)
            contents = tf.image.convert_image_dtype(contents, dtype=tf.float32,saturate=True)
            contents = preprocess(contents)  
            contents = tf.expand_dims(contents, axis=0)            
    return samples, contents

def Optimizer(targets,initials,contents,FLAGS=None):
    # Define the container of the parameter
    Procedure = collections.namedtuple('Procedure', 'optimizer, \
                                        content_loss, style_loss,tv_loss, gen_loss \
                                        outputs, global_step, \
                                        learning_rate')

    # Build the generator part
    with tf.variable_scope('generator'):
        gen_output = generator(FLAGS,targets,initials,contents,reuse=False)

    # Calculating the generator loss
    with tf.name_scope('generator_loss'):   
        with tf.name_scope('tv_loss'):
            tv_loss = total_variation_loss(gen_output)

        with tf.name_scope('style_loss'):
            def gram(features):
                features = tf.reshape(features,[-1,features.shape[3]])
                return tf.matmul(features,features,transpose_a=True)\
                             / tf.cast(features.shape[0]*features.shape[1],dtype=tf.float32)

            _, vgg_gen_output = vgg_19(gen_output,is_training=False,reuse=False)
            _, vgg_tar_output = vgg_19(targets,is_training=False,reuse=True)
            style_layer_list = get_style_layer_list(FLAGS.top_style_layer,False)
            sl = tf.zeros([])
            ratio_list=[100.0, 1.0, 0.1, 0.0001, 1.0, 100.0]
            for i in range(len(style_layer_list)):
                tar_layer = style_layer_list[i]
                target_layer = get_layer_scope(tar_layer)
                gen_feature = vgg_gen_output[target_layer]
                tar_feature = vgg_tar_output[target_layer]
                diff = tf.square(gram(gen_feature)-gram(tar_feature))
                sl = sl + tf.reduce_mean(tf.reduce_sum(diff,axis=0)) * ratio_list[i] 
            style_loss = sl
        
        with tf.name_scope('content_loss'):
            if FLAGS.task_mode == 'style_transfer':
                _, vgg_content_output = vgg_19(contents,is_training=False,reuse=True)
                target_layer = get_layer_scope(FLAGS.content_layer)
                content_feature = vgg_content_output[target_layer]
                gen_feature = vgg_gen_output[target_layer]
                #content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(gen_feature - content_feature), axis =3))
                content_loss = tf.reduce_mean(tf.square(gen_feature - content_feature))
            else:
                content_loss = tf.zeros([])
        
        if FLAGS.task_mode == 'style_transfer':
            gen_loss = style_loss + FLAGS.W_tv * tv_loss + FLAGS.W_content * content_loss 
        elif FLAGS.task_mode == 'texture_synthesis':
            gen_loss = style_loss + FLAGS.W_tv * tv_loss
        else:
            raise ValueError('task_mode should be \'style_transfer\' or \'texture_synthesis\' !')
        #style_loss = tf.zeros([]);tv_loss = tf.zeros([])
        
    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step() 
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,\
                                                   FLAGS.decay_step, FLAGS.decay_rate,\
                                                   staircase = FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)   
        
    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1 = FLAGS.beta1).\
            minimize(gen_loss, global_step = global_step, var_list = gen_tvars)
    
    return Procedure(
        tv_loss = tv_loss,
        style_loss = style_loss,
        content_loss = content_loss,
        gen_loss = gen_loss,
        outputs = gen_output,
        optimizer = optimizer,
        global_step = global_step,
        learning_rate = learning_rate
    )