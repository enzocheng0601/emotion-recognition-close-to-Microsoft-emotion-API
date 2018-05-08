
import os
import os.path

import numpy as np
import tensorflow as tf

import input_data
import VGG
import tools
import inference

import sys



IMG_W = 64
IMG_H = 64
N_CLASSES = 8
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = int(sys.argv[7])
IS_PRETRAIN = True


def train():
    train_log_dir = sys.argv[5]
    val_log_dir = sys.argv[6]
    
    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = input_data.read_image(is_train=True,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=True)
        val_image_batch, val_label_batch = input_data.read_image(
                                                 is_train=False,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=False)
        
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES]) 
    
    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.losses(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = tools.optimizer(loss, learning_rate, my_global_step)   
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()   
       
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    
    #tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])   


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    print('start training....')
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:tra_images, y_:tra_labels})            
            if step % 50 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op, feed_dict ={x: tra_images, y_:tra_labels})
                tra_summary_writer.add_summary(summary_str, step)
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x:val_images,y_:val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))

                summary_str = sess.run(summary_op, feed_dict ={x: tra_images, y_:tra_labels})
                val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
    inference.extract_face()
    train()



