#!/usr/bin/python
__author__ = '@smafjal {afjal.sm19@gmail.com}'

import numpy as np
from random import shuffle
import tensorflow as tf

# parameters
look_back = 20 # time-step
feature_length = 1
num_units = 24 # cell units
model_path="model/final/setbit_count_model.ckpt-500"

def user_test_data():
    x = raw_input("Enter a number (max val: 4096): ")
    if x.strip()=='':
        return []
    x=int(x)
    X='{0:020b}'.format(x)
    print x," on binary:-> ",X
    tmp=[];
    for i in X:tmp.append([i])
    X=np.array(tmp)
    return X

def model_test():

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, look_back, feature_length],name="X-Input")
        Y = tf.placeholder(tf.float32, 21,name="Y-Output")

    with tf.name_scope("weights"):
        W = tf.Variable(tf.truncated_normal([num_units, 21]),name="W")

    with tf.name_scope("biases"):
        B = tf.Variable(tf.constant(0.1, shape=[21]),name="B")

    # model-defination
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
    weights, _ = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
    weights = tf.transpose(weights, [1, 0, 2])
    last_weight = tf.gather(weights, int(weights.get_shape()[0]) - 1) # get the last weight

    with tf.name_scope("softmax"):
        pred = tf.nn.softmax(tf.matmul(last_weight, W) + B)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess,model_path)

        while(True):
            user_x=user_test_data()
            if len(user_x)==0:continue
            pred_val = sess.run(pred,{X:[user_x]})
            ind = np.argmax(pred_val)
            print "---Predicted Value---- \n",pred_val
            print "Number of One: ",ind

def main():
    model_test()

if __name__ == "__main__":
    main()

