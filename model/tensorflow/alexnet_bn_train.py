import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 32
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
step_save = 5000
step_phases = [2000]
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

def gen_features(x, keep_dropout, train_phase):
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

    return features

def domain(x, keep_dropout, train_phase):
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
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)
tasks = ["SceneRecognition", "ObjectRecognition"]

domain_optimizers = [0 for i in range(len(tasks))]
feature_optimizers = [0 for i in range(len(tasks))]
full_optimizers = [0 for i in range(len(tasks))]
accuracy1s = [0 for i in range(len(tasks))]
accuracy5s = [0 for i in range(len(tasks))]
indexes = [0 for i in range(len(tasks))]
losses = [0 for i in range(len(tasks))]
regularization_losses = [0 for i in range(len(tasks))]
evaluation_losses = [0 for i in range(len(tasks))]

features = gen_features(x, keep_dropout, train_phase)
logits_arr = domain(features, keep_dropout, train_phase)
for i in range(len(tasks)):
    task = tasks[i]
    logits = logits_arr[i]
    regularization_losses[i] = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    evaluation_losses[i] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    losses[i] = evaluation_losses[i] + regularization_losses[i]
    domain_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = task)
    feature_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "EntryFlow") + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "MiddleFlow")
    domain_optimizers[i] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(losses[i], var_list = domain_list)
    feature_optimizers[i] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(losses[i], var_list = feature_list)
    full_optimizers[i] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(losses[i])

    values, indexes[i] = tf.nn.top_k(logits, 1)
    accuracy1s[i] = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
    accuracy5s[i] = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()
restorer = saver
#restorer = tf.train.Saver(val_list)

def initialize(sess):
    sess.run(init)            
    if len(start_from)>1:        
        restorer.restore(sess, start_from)

def train(sess):
    step = 0
    phase_schedule = ['full']#'feature', 'domain']
    phases = {'feature' : 0, 'domain' : 1, 'full' : 2}
    loader_trains = [scene_loader_train, obj_loader_train]
    loader_vals = [scene_loader_val, obj_loader_val]
    loader_tests = [scene_loader_test, obj_loader_test]
    optimizers = [domain_optimizers, feature_optimizers, full_optimizers]
    phase_index = 0
    step_phase = step_phases[phase_index]
    
    while step < training_iters:
        if step % step_phase < step_phase/2:
            task_index = 0
        else:
            task_index = 1
        i = task_index
        loader_train = loader_trains[task_index]
        loader_val = loader_vals[task_index]
        images_batch, labels_batch = loader_train.next_batch(batch_size)

        if step % step_phase == 0:
            phase_index += 1
            if phase_index == len(phase_schedule):
                phase_index = 0
            phase = phases[phase_schedule[phase_index]]
            step_phase = step_phases[phase_index]

        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            l, acc1, acc5, reg, eval_loss = sess.run([losses[i], accuracy1s[i], accuracy5s[i], regularization_losses[i], evaluation_losses[i]],
                                                     feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Phase " + phase_schedule[phase_index] + ", task " + tasks[task_index] +", Training Loss= " + "{:.6f}".format(l) + ", Accuracy Top1 = " + "{:.4f}".format(acc1) +
                  ", Top5 = " + "{:.4f}".format(acc5) + ", Evaluation Loss = " + "{:.4f}".format(eval_loss) + ", Regularization Loss = " + "{:.4f}".format(reg))
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([losses[i], accuracy1s[i], accuracy5s[i]], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Validation Loss= " + "{:.6f}".format(l) + ", Accuracy Top1 = " + "{:.4f}".format(acc1) + ", Top5 = " + "{:.4f}".format(acc5))
            
        sess.run(optimizers[phase][task_index], feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
            
        step += 1
                
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)            
            print("Model saved at Iter %d !" %(step))
            for i in range(len(tasks)):
                print("Task:" + str(tasks[i]))
                validate(sess, loader_vals[i], i)
#                evaluate(sess, loader_tests[i], i)
            
    print("Optimization Finished!")

def validate(sess, loader_val, task_index):
    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    
    i = 0
    while i < loader_val.size():
        if i + batch_size < loader_val.size():
            size = batch_size
        else:
            size = loader_val.size() - i
        i += size
        images_batch, labels_batch = loader_val.next_batch(size)    
        acc1, acc5 = sess.run([accuracy1s[task_index], accuracy5s[task_index]], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1 * size
        acc5_total += acc5 * size
    acc1_total /= loader_val.size()
    acc5_total /= loader_val.size()
    print('Validation Finished! ', 'Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))

def evaluate(sess, loader_test, task_index):
    print('Evaluation on the whole test set...')
    predictions = []
    i = 0
    while i < loader_test.size():
        if i + batch_size < loader_test.size():
            size = batch_size
        else:
            size = loader_test.size() - i
        i += size
        images_batch, labels_batch = loader_test.next_batch(size)    
        preds = sess.run(indexes[task_index], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        predictions = predictions + [x[0] for x in preds]
    print(predictions)    
    
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
