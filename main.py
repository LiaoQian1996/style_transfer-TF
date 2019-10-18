from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import data_loader, generator 
from lib.ops import *
import math
import time
import numpy as np
import scipy.misc
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

parser = argparse.ArgumentParser()

# Input Arguments
parser.add_argument(
    '--task_mode',
    help = 'style transfer or texture synthesis',
    default = 'texture_synthesis',
    choices = ['style_transfer','texture_synthesis']
)

parser.add_argument(
    '--output_dir',
    help = 'output dictionary',
    required = True
)

parser.add_argument(
    '--vgg_ckpt',
    help = 'checkpoint of vgg networks, the check point file of pretrained model should be downloaded',
    default = '/home/liaoqian/DATA/vgg19/vgg_19.ckpt'
)

parser.add_argument(
    '--target_dir',
    help = 'path of target img, texture sample image or style image',
    default = './imgs/tomato.png' 
)

parser.add_argument(
    '--content_dir',
    help = 'path of the content image, come into force only in style_transfer task_mode',
    default = './imgs/pepper.png'
)

parser.add_argument(
    '--initials',
    help = 'initialized mode of synthesis, come into force only in style_transfer task_mode',
    choices = ['noise', 'content', 'style'],
    default = 'content'
)

parser.add_argument(
    '--top_style_layer',
    help = 'the top layer of vgg network layers used to compute style_loss',
    default = 'VGG54',
    choices = ['VGG11','VGG21','VGG31','VGG41','VGG51','VGG54']
)

parser.add_argument(
    '--content_layer',
    help = 'use which layer of content image to transfer style',
    default = 'VGG41'
)

parser.add_argument(
    '--texture_shape',
    help = 'img_size of synthesis output texture, if set to [-1,-1], the shape will be \
    the same as sample texture image',
    nargs = '+',
    type = int
)

parser.add_argument(
    '--stddev',
    help = 'standard deviation of noise that add to initial content image',
    type = float,
    default = 1.0
)

parser.add_argument(
    '--W_tv',
    help = 'weight of total variation loss',
    type = float,
    default = 0.1
)
parser.add_argument(
    '--W_content',
    help = 'weight of content loss',
    type = float,
    default = 1.0
)

parser.add_argument(
    '--max_iter',
    help = 'max iteration',
    type = int,
    default = 100,
    required = True
)

parser.add_argument(
    '--save_freq',
    help = 'frequency of save',
    type = int,
    default = 1000
)

FLAGS = parser.parse_args()
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
    
targets, initials = data_loader(FLAGS)  

# how to initialize the synthesis image
if FLAGS.task_mode == 'texture_synthesis': 
    if FLAGS.initials == 'noise':
        initials = None
    elif FLAGS.initials == 'content':
        if FLAGS.content_dir is None:
            raise ValueError('The content image path is needed')
elif FLAGS.task_mode == 'style_transfer':
    contents = initials
    if FLAGS.initials == 'content':
        initials = contents
    elif FLAGS.initials == 'noise':
        initials = None
    elif FLAGS.initials == 'style':
        initials = targets
        
with tf.variable_scope('generator'):
    if FLAGS.task_mode == 'texture_synthesis': 
        gen_output = generator(FLAGS, targets, initials, None, reuse = False)
    elif FLAGS.task_mode == 'style_transfer':
        gen_output = generator(FLAGS, targets, initials, contents, reuse = False)

    # Calculating the generator loss
with tf.name_scope('generator_loss'):   
    with tf.name_scope('tv_loss'):
        tv_loss = total_variation_loss(gen_output)

    with tf.name_scope('style_loss'):
        _, vgg_gen_output = vgg_19(gen_output,is_training=False,reuse=False)
        _, vgg_tar_output = vgg_19(targets,is_training=False,reuse=True)
        style_layer_list = get_layer_list(FLAGS.top_style_layer,False)
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
    gen_loss = 1e6 * gen_loss

with tf.name_scope('generator_train'):
    gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        gen_loss, var_list = gen_tvars, method='L-BFGS-B',
        options = {'maxiter': FLAGS.max_iter, 'disp': True})

print('Finish building the Optimizer !!!')

outputs = gen_output

with tf.name_scope('converted_images'):
    outputs = deprocess(outputs)
    converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
vgg_restore = tf.train.Saver(vgg_var_list)
  
# Start the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def print_loss(gl, sl, cl, tvl):
    print('gen_loss : %s' % gl )
    print('style_loss : %s' % sl )
    print('content_loss : %s' % cl ) 
    print('tv_loss : %s' % tvl )
    
init_op = tf.global_variables_initializer()   
with tf.Session(config = config) as sess:
    sess.run(init_op)
    vgg_restore.restore(sess, FLAGS.vgg_ckpt)
    print('VGG19 restored successfully!!')
    print('Optimization starts!!!')
    start = time.time()
    optimizer.minimize(sess, loss_callback = print_loss,
                     fetches = [gen_loss, style_loss, content_loss, tv_loss])
    gen_output = np.squeeze(gen_output.eval())
    
    tar_name = FLAGS.target_dir.split('/')[-1]; tar_name = tar_name.split('.')[0]
    content_name = FLAGS.content_dir.split('/')[-1]; content_name = content_name.split('.')[0]
    total_time = (time.time() - start)
    if FLAGS.task_mode == 'style_transfer':
        scipy.misc.toimage(gen_output, cmin=-1.0, cmax=1.0) \
                        .save(FLAGS.output_dir + '%s_%.4e_%.4e_%.1f_%s_%s.png'
                              %(FLAGS.top_style_layer,style_loss.eval(),content_loss.eval(),\
                                total_time,tar_name,content_name))
    else:
        scipy.misc.toimage(gen_output, cmin=-1.0, cmax=1.0) \
                        .save(FLAGS.output_dir + '%s_%.4e_%.1f_%s.png'
                              %(FLAGS.top_style_layer,style_loss.eval(),\
                                total_time, tar_name))
    print("Style image size :        %d   %d" %(targets.shape[1], targets.shape[2]))
    if FLAGS.task_mode == 'style_transfer':
        print("Content image size :      %d   %d" %(contents.shape[1], targets.shape[2]))
    else:
        print("Output image size :       %d   %d" %(gen_output.shape[0], gen_output.shape[1]))
    print("Total synthesizing time : %.1f sec" % total_time)
    print('Optimization done ! ')        

    
    
    