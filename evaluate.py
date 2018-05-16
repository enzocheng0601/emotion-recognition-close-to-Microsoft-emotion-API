import math
import os
import os.path
import numpy as np
import tensorflow as tf
import input_data
import VGG
import tools
import inference
import sys
import matplotlib.pyplot as plt

def evaluate():
    with tf.Graph().as_default():

        log_dir = sys.argv[1]
        raw_test_dir = sys.argv[2]
        test_dir = sys.argv[3]
        test_json_file_dir = sys.argv[4]
        emotion_api_dir = sys.argv[5]
        n_test = 0
        N_CLASSES = 8
        IS_PRETRAIN = False
        images, labels = input_data.read_test_image(raw_test_dir, test_dir, test_json_file_dir, emotion_api_dir)
        for item in os.listdir(test_dir):
            n_test += 1

        logits = VGG.VGG16N(images, N_CLASSES, IS_PRETRAIN)
        log = tf.nn.softmax(logits)
        correct = tools.num_correct_prediction(log, labels)
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                print('\nEvaluating......')
                valid_test = len(sess.run(labels))
                BATCH_SIZE = valid_test
                num_step = int(math.floor(valid_test / BATCH_SIZE))
                num_sample = num_step*BATCH_SIZE
                step = 0
                total_correct = 0
                
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' %num_sample)
                print('Valid testing samples: %d' %valid_test)
                print('Total correct predictions: %d' %total_correct)
                print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == "__main__":
    evaluate()
