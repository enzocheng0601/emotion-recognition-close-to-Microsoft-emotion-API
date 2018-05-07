
import tensorflow as tf
import numpy as np
import csv
import os
import cv2
import json
import data_processing
import sys


#%% Reading data

def read_image(is_train, batch_size, shuffle):
    """Read CIFAR10
    
    Args:
        data_dir: the directory of CIFAR10
        is_train: boolen
        batch_size:
        shuffle:       
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32
    
    """
    img_width = 64
    img_height = 64
    img_depth = 3
    label_bytes = 1
    image_bytes = img_width*img_height*img_depth
    
    
    with tf.name_scope('input'):
        
        if is_train:
            print('preparing data for training...')
            index = 'training'
            filenames = []
            label = []
            data_processing.get_label()
            filenames, label = data_processing.labeling(filenames, label, index)
        else:
            print('preparing data for validation...')
            index = 'validation'
            filenames = []
            label = []
            filenames, label = data_processing.labeling(filenames, label, index)
        #tmp = np.array([filenames, label])
        #tmp = tmp.transpose()
        #np.random.shuffle(tmp)
#        # data argumentation

#        image = tf.random_crop(image, [24, 24, 3])# randomly crop the image size to 24 x 24
#        image = tf.image.random_flip_left_right(image)
#        image = tf.image.random_brightness(image, max_delta=63)
#        image = tf.image.random_contrast(image,lower=0.2,upper=1.8)
        filenames_list = tf.cast(filenames, tf.string)
        label_list = tf.cast(label, tf.float32)
        input_Queue = tf.train.slice_input_producer([filenames_list, label_list])
        

        label = input_Queue[1]

        image_undecoded = tf.read_file(input_Queue[0])
        image = tf.image.decode_jpeg(image_undecoded, channels = 3)
        image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)

        image = tf.image.per_image_standardization(image) #substract off the mean and divide by the variance 

        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                                    [image, label], 
                                    batch_size = batch_size,
                                    num_threads= 64,
                                    capacity = 20000,
                                    min_after_dequeue = 100)
        else:
            images, label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size = batch_size,
                                    num_threads = 64,
                                    capacity= 2000)
        return images, label_batch
#%%






