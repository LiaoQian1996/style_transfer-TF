from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import data_loader, Optimizer
from lib.ops import *
import math
import time
import numpy as np
import scipy.misc
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
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
    '--summary_dir',
    help = 'summary dictionary',
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
    default = 'noise'
)
    
parser.add_argument(
    '--clip_output',
    help = 'if enable, range of gen_img will be forced cliped into [-1,1]',
    type = bool,
    default = True 
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
    '--learning_rate',
    help = 'learning rate in the beginning of optimization',
    type = float,
    default = 0.1
)
parser.add_argument(
    '--beta1',
    type = float,
    default = 0.9
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
    '--decay_step',
    help = 'in which step learning will decay to decaty_rate',
    type = int,
    default = 10000, 
)
parser.add_argument(    
    '--decay_rate',
    help = 'the decay ratio of learning rate',
    type = float,
    default = 0.1 
)
parser.add_argument(
    '--stair',
    help = 'learning rate decay in stair or in exponential way smoothly',
    type = bool,
    default = False 
)
parser.add_argument(
    '--max_iter',
    help = 'max iteration',
    type = int,
    default = 10000,
    required = True
)
parser.add_argument(
    '--display_freq',
    help = 'frequency of display optimization infomation',
    type = int,
    default = 20
)
parser.add_argument(
    '--summary_freq',
    help = 'frequency of summary',
    type = int,    
    default = 100
)
parser.add_argument(
    '--save_freq',
    help = 'frequency of save',
    type = int,
    default = 1000
)

parser.add_argument(
    '--save_steps',
    help = 'steps of save output, notice that the steps must be multiple of display_freq',
    nargs = '+',
    type = int,
    default = [20,100,500]
)

FLAGS = parser.parse_args()
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

if FLAGS.summary_dir is None:
    raise ValueError('The summary directory is needed')
    
# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# Load data for training and testing
# ToDo Add online downscaling

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

if FLAGS.task_mode == 'texture_synthesis': 
    Optimizer = Optimizer(targets, initials, None, FLAGS)
elif FLAGS.task_mode == 'style_transfer':
    Optimizer = Optimizer(targets, initials, contents, FLAGS)

print('Finish building the Optimizer !!!')

outputs = Optimizer.outputs

with tf.name_scope('converted_images'):
    outputs = deprocess(outputs)
    targets = deprocess(targets)
    converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)
    converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)

tf.summary.scalar('style_loss',Optimizer.style_loss)
tf.summary.scalar('tv_loss',Optimizer.tv_loss)
tf.summary.scalar('gen_loss',Optimizer.gen_loss)
tf.summary.histogram('outputs',Optimizer.outputs)
tf.summary.histogram('converted_outputs',converted_outputs)
tf.summary.histogram('converted_targets',converted_targets)
tf.summary.image('converted_outputs',converted_outputs)

var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
weight_initiallizer = tf.train.Saver(var_list)
vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
vgg_restore = tf.train.Saver(vgg_var_list)
    
# Start the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Use superviser to coordinate all queue and summary writer
sv = tf.train.Supervisor(logdir = FLAGS.summary_dir, save_summaries_secs = 0, saver = None)
    
with sv.managed_session(config = config) as sess:
    vgg_restore.restore(sess, FLAGS.vgg_ckpt)
    print('VGG19 restored successfully!!')
    print('Optimization starts!!!')
    start = time.time()
    max_iter = FLAGS.max_iter
    save_step = FLAGS.save_steps
    
    optimizer = Optimizer.optimizer
    optimizer.minimize(sess)
    
    fetches = {}
    fetches["Net.global_step"] = Optimizer.global_step
    fetches["tv_loss"] = Optimizer.tv_loss
    fetches["style_loss"] = Optimizer.style_loss
    fetches["learning_rate"] = Optimizer.learning_rate
    if FLAGS.task_mode == 'style_transfer':
        fetches["content_loss"]= Optimizer.content_loss

    fetches["summary"] = sv.summary_op
    fetches["outputs"] = converted_outputs   

    results = sess.run(fetches)
    print(results["content_loss"])
    print(results["style_loss"])
    print(results["tv_loss"])
    
    '''
    if ((step + 1) % FLAGS.summary_freq) == 0:
        print('Recording summary!!')
        sv.summary_writer.add_summary(results['summary'], (step + 1))

    if ((step + 1) % FLAGS.display_freq) == 0:
        rate = (step + 1) * 1 / (time.time() - start)
        remaining = (max_iter - step) * 1 / rate
        print("current style image : %s  remaining %dm " % \
              (FLAGS.target_dir.split('/')[-1], remaining / 60))
        if FLAGS.task_mode == 'style_transfer':
            print("current content image : ",FLAGS.content_dir.split('/')[-1])
        #print("progress  image/sec %0.1f  remaining %dm" % (rate, remaining / 60))
        print("global_step", results["Net.global_step"])
        print("tv_loss", results["tv_loss"])
        if FLAGS.task_mode == 'style_transfer':
            print("content_loss",results["content_loss"])
        print("style_loss", results["style_loss"])           
        print("learning_rate", results['learning_rate'])        
    '''          
    
    step = FLAGS.max_iter - 1    
    if FLAGS.task_mode == 'texture_synthesis':
        img_name = FLAGS.target_dir.split('/')[-1]
        img_name = img_name.split('.')[0]
        im = np.squeeze(results["outputs"])
        scipy.misc.imsave(FLAGS.output_dir +'%s_%i_%.4e_%.4e_%s.png'%(FLAGS.top_style_layer,step+1,results["style_loss"],results["tv_loss"],img_name), im)

    elif FLAGS.task_mode == 'style_transfer':
        tar_name = FLAGS.target_dir.split('/')[-1]
        tar_name = tar_name.split('.')[0]
        content_name = FLAGS.content_dir.split('/')[-1]
        content_name = content_name.split('.')[0]
        im = np.squeeze(results["outputs"])
        scipy.misc.imsave(FLAGS.output_dir +'%s_%i_%.4e_%.4e_%s_%s.png'%(FLAGS.top_style_layer,step+1,results["style_loss"],results["content_loss"],tar_name,content_name), im)          
    print('Optimization done !!! ')        
