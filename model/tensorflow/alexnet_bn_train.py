import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 64
load_size = 128
fine_size = 112
c = 3
data_mean = np.load("./training_mean.npy")
experiment = 'Xception'

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
step_display = 50
step_save = 5000
path_save = './models/alexnet_bn'
start_from = ''
regularization_scale = 0.00001
regularizer = tf.contrib.layers.l2_regularizer(regularization_scale);

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def up_sample(x, keep_dropout, train_phase, name):
    with tf.name_scope("Upwnsample_"+name):    
        depth = x.get_shape().as_list()[3]
        up = tf.nn.separable_conv2d(x, tf.get_variable('wud' + name, shape = [3, 3, depth, 1], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                    tf.get_variable('wup' + name, shape = [1, 1, depth, depth*2], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                    padding = 'SAME', strides = [1,1,1,1])
        up = batch_norm_layer(up, train_phase, 'bu' + name)
        up = tf.nn.relu(up)
        return up

def down_sample(x, keep_dropout, train_phase, name):
    with tf.name_scope("Downsample_"+name):
        depth = x.get_shape().as_list()[3]
        dl = tf.nn.conv2d(x, tf.get_variable('wdl' + name, shape = [3, 3, depth, depth*2], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                          padding = 'SAME', strides = [1,2,2,1])
        dl = batch_norm_layer(dl, train_phase, 'bdl'+ name)
        dl = tf.nn.relu(dl)
        
        x = tf.nn.relu(x)
        dr = tf.nn.separable_conv2d(x, tf.get_variable('wdrdf' + name, shape = [3, 3, depth, 1], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                    tf.get_variable('wdrpf' + name, shape = [1, 1, depth, depth*2], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                    padding = 'SAME', strides = [1,1,1,1])
        dr = batch_norm_layer(dr, train_phase, 'bdrf'+ name)
        dr = tf.nn.relu(dr)
        dr = tf.nn.separable_conv2d(dr, tf.get_variable('wdrds' + name, shape = [3, 3, depth*2, 1], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                    tf.get_variable('wdrps' + name, shape = [1, 1, depth*2, depth*2], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                    padding = 'SAME', strides = [1,1,1,1])
        dr = batch_norm_layer(dr, train_phase, 'bdrs'+ name)    
        dr = tf.nn.max_pool(dr, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')            
        return tf.add(dl, dr)

def feature_select(x, keep_dropout, train_phase, id):
    with tf.name_scope("FeatureSelect_" + str(id)):
        depth = x.get_shape().as_list()[3]    
        m = x
        for k in range(3):
            name = str(k) + id        
            m = tf.nn.relu(m)
            m = tf.nn.separable_conv2d(m, tf.get_variable('mdf' + name, shape = [3, 3, depth, 1], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                       tf.get_variable('mpf' + name, shape = [1, 1, depth, depth], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                       padding = 'SAME', strides = [1,1,1,1])    
            m = batch_norm_layer(m, train_phase, 'bmf'+ name)        
        return tf.add(x, m)

def alexnet(x, keep_dropout, train_phase):
    with tf.name_scope("EntryFlow"):
        e1 = tf.nn.conv2d(x, tf.get_variable('we1', shape = [3, 3, 3, 32], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer), padding = 'SAME', strides = [1,2,2,1])
        e1 = batch_norm_layer(e1, train_phase, 'be1')
        e1 = tf.nn.relu(e1)

        e2 = tf.nn.conv2d(e1, tf.get_variable('we2', shape = [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer), padding = 'SAME', strides = [1,1,1,1])
        e2 = batch_norm_layer(e2, train_phase, 'be2')
        e2 = tf.nn.relu(e2)

        e3 = down_sample(e2, keep_dropout, train_phase, str(3))
        e4 = down_sample(e3, keep_dropout, train_phase, str(4))

    with tf.name_scope("MiddleFlow"):
        for k in range(8):
            e4 = feature_select(e4, keep_dropout, train_phase, str(k))

    with tf.name_scope("ExitFlow"):
        e5 = down_sample(e4, keep_dropout, train_phase, str(5))
        e6 = up_sample(e5, keep_dropout, train_phase, str(6))
        e7 = up_sample(e6, keep_dropout, train_phase, str(7))
        e7 = tf.reduce_mean(e7, axis = [1,2])        
        return e7

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = alexnet(x, keep_dropout, train_phase)

# Define loss and optimizer
regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
tf.summary.scalar('Regularization Loss', regularization_loss)
evaluation_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
tf.summary.scalar('Evaluation Loss', evaluation_loss)
loss = evaluation_loss + regularization_loss
tf.summary.scalar('Loss', loss)
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))
tf.summary.scalar('Top 1 Accuracy', accuracy1)
tf.summary.scalar('Top 5 Accuracy', accuracy5)
summary = tf.summary.merge_all()

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
train_writer = tf.summary.FileWriter('logs/' + experiment + 'train', tf.get_default_graph());
val_writer = tf.summary.FileWriter('logs/' + experiment + 'val', tf.get_default_graph());

# Launch the graph
with tf.Session() as sess:
    
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)
    
    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5, reg, eval_loss, log = sess.run([loss, accuracy1, accuracy5, regularization_loss, evaluation_loss, summary],
                                                          feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) +
                  ", Training Loss= " + "{:.6f}".format(l) +
                  ", Accuracy Top1 = " + "{:.4f}".format(acc1) +
                  ", Top5 = " + "{:.4f}".format(acc5) +
                  ", Evaluation Loss = " + "{:.4f}".format(eval_loss) +
                  ", Regularization Loss = " + "{:.4f}".format(reg))
            train_writer.add_summary(log, step)

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5, log = sess.run([loss, accuracy1, accuracy5, summary], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))
            val_writer.add_summary(log, step)
        
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))

    train_writer.close()
    val_writer.close()
    print("Optimization Finished!")


    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
