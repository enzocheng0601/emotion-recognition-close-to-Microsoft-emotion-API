
import tensorflow as tf
import numpy as np
import csv
import os
import cv2
import json
import data_processing
import sys
import inference
import subprocess
import time


#%% Reading data

def read_image(is_train, batch_size, shuffle):
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


def read_test_image(test_dir, test_output_dir, test_json_output_dir, emotion_api_path):
    filenames = []
    label = []
    inference.extract_face_for_test(test_dir, test_output_dir)
    if not os.path.exists(test_json_output_dir):
        os.mkdir(test_json_output_dir)
    print(' getting label from MS emotion api and labeling...')
    counter = 1
    for image_name in os.listdir(test_output_dir):
        image = cv2.imread(test_output_dir + image_name)
        number = image_name.split('.')[0]
        p = subprocess.Popen([emotion_api_path, test_output_dir+image_name, test_json_output_dir+str(number)+'.json'])
        p.wait()
        json_data = open(test_json_output_dir+str(number)+'.json').read()
        data = json.loads(json_data)
        if len(data['arr']) > 0:
            faceRectangle = data['arr'][0]['faceRectangle']
            faceAttribute = data['arr'][0]['faceAttributes']
            (x, y, w, h) = (faceRectangle['left'], faceRectangle['top'], faceRectangle['width'], faceRectangle['height'])
            emotion = faceAttribute['emotion']
            label_array = []
            print('ghhhhhhhh')
            for key in sorted(emotion.keys()):
                label_array.append(emotion[key])
            print(label_array)
            roi_cropped = image[y:y + h, x: x + w]

            cv2.imwrite(test_output_dir + str(counter) + '.jpg', roi_cropped)
            label.append(label_array)
            filenames.append(test_output_dir + str(counter) + '.jpg')
            counter += 1
    print('finished')
    time.sleep(1)
    #os.system('cls')

    print('preparing for testing')
    batch_size = counter - 1
    filenames_list = tf.cast(filenames, tf.string)
    label_list = tf.cast(label, tf.float32)
    input_Queue = tf.train.slice_input_producer([filenames_list, label_list])
    label = input_Queue[1]
    image_undecoded = tf.read_file(input_Queue[0])
    image = tf.image.decode_jpeg(image_undecoded, channels = 3)
    image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)
    image = tf.image.per_image_standardization(image)
    images, label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size = batch_size,
                                    num_threads = 64,
                                    capacity= 2000)
    return images, label_batch






