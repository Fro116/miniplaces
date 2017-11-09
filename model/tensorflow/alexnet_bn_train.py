import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 22
load_size = 128
fine_size = 112
c = 3
data_mean = np.load("./training_mean.npy")
experiment = 'binary'

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 500000
step_display = 50
step_save = 7000
path_save = './models/alexnet_bn'
start_from = ''#./models/xception/alexnet_bn-50000'
regularization_scale = 0.#0.00001
regularizer = tf.contrib.layers.l2_regularizer(regularization_scale);

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=train_phase, reuse=None, trainable=True, scope=scope_bn)

def separable_conv(x, output_depth, name, relu_on_entry):
    with tf.variable_scope("-separable_conv"+name):    
        depth = x.get_shape().as_list()[3]
        conv = x
        if relu_on_entry:
            conv = tf.nn.relu(conv)        
        conv = tf.nn.separable_conv2d(conv, tf.get_variable('-depthwise', shape = [3, 3, depth, 1], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                      tf.get_variable('-pointwise' + name, shape = [1, 1, depth, output_depth], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                                      padding = 'SAME', strides = [1,1,1,1])
        conv = batch_norm_layer(conv, train_phase, '-batch_norm' + name)
        if not relu_on_entry:
            conv = tf.nn.relu(conv)
        return conv

def conv(x, output_depth, filter_size, stride, name, relu):
    with tf.variable_scope("-conv"+name):        
        depth = x.get_shape().as_list()[3]
        conv = tf.nn.conv2d(x, tf.get_variable('-kernel' + name, shape = [filter_size, filter_size, depth, output_depth], initializer = tf.contrib.layers.xavier_initializer(), regularizer = regularizer),
                        padding = 'SAME', strides = [1, stride, stride, 1])
        conv = batch_norm_layer(conv, train_phase, '-batch_norm'+ name)
        if relu:
            conv = tf.nn.relu(conv)
        return conv
    
def down_sample(x, first_depth, second_depth, stride, keep_dropout, train_phase, name):
    with tf.variable_scope("-down_sample"+name):
        bias = conv(x, output_depth = second_depth, filter_size = 1, stride = stride, name = "-bias", relu = False)
        first = separable_conv(x, output_depth = first_depth, name = "-first", relu_on_entry = True)
        second = separable_conv(first, output_depth = second_depth, name = "-second", relu_on_entry = True)
        pool = tf.nn.max_pool(second, ksize=[1, 3, 3, 1], strides=[1, stride, stride, 1], padding='SAME')            
        return tf.add(bias, pool)

def feature_select(x, keep_dropout, train_phase, name):
    with tf.variable_scope("-feature_select" + name):
        depth = x.get_shape().as_list()[3]    
        feature = x
        for k in range(3):
            feature = separable_conv(feature, output_depth = depth, name = "-feature_"+str(k), relu_on_entry = True)
        return tf.add(x, feature)

def network(x, keep_dropout, train_phase):
    with tf.variable_scope("EntryFlow"):
        step1 = conv(x, output_depth = 32, filter_size = 3, stride = 2, relu = True, name = "-1")
        step2 = conv(step1, output_depth = 64, filter_size = 3, stride = 1, relu = True, name = "-2")        
        step3 = down_sample(step2, first_depth = 128, second_depth = 128, stride = 2, keep_dropout = keep_dropout, train_phase = train_phase, name = "-3")
        step4 = down_sample(step3, first_depth = 256, second_depth = 256, stride = 2, keep_dropout = keep_dropout, train_phase = train_phase, name = "-4")
        step5 = down_sample(step4, first_depth = 728, second_depth = 728, stride = 1, keep_dropout = keep_dropout, train_phase = train_phase, name = "-5")

    with tf.variable_scope("MiddleFlow"):
        features = step5
        for k in range(8):
            features = feature_select(features, keep_dropout, train_phase, "-"+str(k))

    with tf.variable_scope("ObjectRecognition"):
        domain1 = down_sample(features, first_depth = 728, second_depth = 1024, stride = 1, keep_dropout = keep_dropout, train_phase = train_phase, name = "-1")
        domain2 = separable_conv(domain1, output_depth = 1536, name = "-2", relu_on_entry = False)
        domain3 = separable_conv(domain2, output_depth = 2048, name = "-3", relu_on_entry = False)
        domain4 = separable_conv(domain3, output_depth = 175, name = "-4", relu_on_entry = False)
        objects = tf.reduce_mean(domain4, axis = [1,2])

    with tf.variable_scope("SceneRecognition"):
        domain1 = down_sample(features, first_depth = 728, second_depth = 1024, stride = 1, keep_dropout = keep_dropout, train_phase = train_phase, name = "-1")
        domain2 = separable_conv(domain1, output_depth = 1536, name = "-2", relu_on_entry = False)
        domain3 = separable_conv(domain2, output_depth = 2048, name = "-3", relu_on_entry = False)
        domain4 = separable_conv(domain3, output_depth = 100, name = "-4", relu_on_entry = False)
        scenes = tf.reduce_mean(domain4, axis = [1,2])

    return [scenes, objects]
    
# tf Graph input
x1 = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
x2 = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y1 = tf.placeholder(tf.int64, None)
y2 = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)
x = tf.concat([x1, x2], axis = 0)

logits = network(x, keep_dropout, train_phase)
logits1 = logits[0][0:batch_size]
logits2 = logits[1][batch_size:(2*batch_size)]
loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1)) +
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2)))
accuracy1 = [tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits1, y1, 1), tf.float32)), tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits2, y2, 1), tf.float32))]
accuracy5 = [tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits1, y1, 5), tf.float32)), tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits2, y2, 5), tf.float32))]
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()
restorer = saver

def initialize(sess):
    sess.run(init)            
    if len(start_from)>1:        
        restorer.restore(sess, start_from)

def train(sess):
    step = 0
    
    while step < training_iters:
        images_batch1, labels_batch1 = scene_loader_train.next_batch(batch_size)
        images_batch2, labels_batch2 = obj_loader_train.next_batch(batch_size)                

        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x1: images_batch1, x2: images_batch2, y1: labels_batch1, y2: labels_batch2, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Training Loss= " + str(l) + ", Accuracy Top1 = " + str(acc1) + ", Top5 = " + str(acc5))
            val_images_batch1, val_labels_batch1 = scene_loader_val.next_batch(batch_size)
            val_images_batch2, val_labels_batch2 = obj_loader_val.next_batch(batch_size)
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5],
                                                     feed_dict={x1: val_images_batch1, x2: val_images_batch2, y1: val_labels_batch1, y2: val_labels_batch2, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Validation Loss= " + str(l) + ", Accuracy Top1 = " + str(acc1) + ", Top5 = " + str(acc5))
            
        sess.run(train_optimizer, feed_dict={x1: images_batch1, x2: images_batch2, y1: labels_batch1, y2: labels_batch2, keep_dropout: dropout, train_phase: True})
            
        step += 1
                
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)            
            print("Model saved at Iter %d !" %(step))
            validate(sess)
            #evaluate(sess)
            
    print("Optimization Finished!")

def validate(sess):
    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    acc1_total = np.array([0., 0.])
    acc5_total = np.array([0., 0.])
    scene_loader_val.reset()
    obj_loader_val.reset()    
    
    i = 0
    while i < scene_loader_val.size():
        if i + batch_size < scene_loader_val.size():
            size = batch_size
        else:
            size = scene_loader_val.size() - i
        i += size
        val_images_batch1, val_labels_batch1 = scene_loader_val.next_batch(batch_size)
        val_images_batch2, val_labels_batch2 = obj_loader_val.next_batch(batch_size)        
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x1: val_images_batch1, x2: val_images_batch2, y1: val_labels_batch1, y2: val_labels_batch2, keep_dropout: 1., train_phase: False})
        acc1_total += np.array(acc1) * size
        acc5_total += np.array(acc5) * size
        
    acc1_total /= scene_loader_val.size()
    acc5_total /= scene_loader_val.size()
    print('Validation Finished! ', 'Accuracy Top1 = ' + str(acc1_total) + ", Top5 = " + str(acc5_total))

# def evaluate(sess, loader_test, task_index):
#     print('Evaluation on the whole test set...')
#     predictions = []
#     i = 0
#     while i < loader_test.size():
#         if i + batch_size < loader_test.size():
#             size = batch_size
#         else:
#             size = loader_test.size() - i
#         i += size
#         images_batch, labels_batch = loader_test.next_batch(size)    
#         preds = sess.run(indexes[task_index], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
#         predictions = predictions + [x[0] for x in preds]
#     print(predictions)    
    
def train_network():
    # Launch the graph
    with tf.Session() as sess:        
        initialize(sess)
        train(sess)

opt_data_train = {
    'data_root': '../../data/images/',
    'data_list': '../../data/obj_train.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'phase': 'training',
}
opt_data_val = {
    'data_root': '../../data/images/',
    'data_list': '../../data/obj_val.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'phase': 'validation',
}
opt_data_test = {
    'data_root': '../../data/images/',
    'data_list': '../../data/obj_val.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'phase': 'evaluation',
}        
obj_loader_train = DataLoaderDisk(**opt_data_train)
obj_loader_val = DataLoaderDisk(**opt_data_val)
obj_loader_test = DataLoaderDisk(**opt_data_test)
scene_loader_train = DataLoaderH5(**opt_data_train)
scene_loader_val = DataLoaderH5(**opt_data_val)
scene_loader_test = DataLoaderH5(**opt_data_test)        
train_network()
