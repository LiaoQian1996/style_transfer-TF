from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import os
import math
from PIL import Image
import numpy as np
    
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