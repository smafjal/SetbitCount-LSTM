#!/usr/bin/python
__author__ = '@smafjal {afjal.sm19@gmail.com}'

import numpy as np
from random import shuffle
import tensorflow as tf

# reset everything to rerun in jupyter
tf.reset_default_graph()
logs_path="logs/"

# parameters
look_back = 20 # time-step
feature_length = 1
num_units = 24 # cell units

def generate_data(limit=10):
    X = ['{0:020b}'.format(i) for i in range(2 ** limit)] # binary convert
    shuffle(X)
    X = [map(int, i) for i in X]

    xlist = []
    for i in X:
        tmp = []
        for j in i:
            tmp.append([j])
        xlist.append(np.array(tmp))
    X = xlist

    Y = []
    for i in X:
        cnt = 0
        for j in i:
            if j[0] == 1:
                cnt+=1
        tmp = ([0] * 21)
        tmp[cnt] = 1 # hoy-vector conversion
        Y.append(tmp)
    return X,Y

def my_lstm(dataX=[],dataY=[]):
    xlen=len(dataX)
    train_limit=(xlen*90)//100

    print "Total input sample: ",xlen
    print "Taken for Train: ",train_limit

    x_train=dataX[:train_limit]
    y_train=dataY[:train_limit]
    x_test=dataX[train_limit:]
    y_test=dataY[train_limit:]

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, look_back, feature_length],name="X-Input")
        Y = tf.placeholder(tf.float32, [None, len(dataY[0])],name="Y-Output")

    with tf.name_scope("weights"):
        W = tf.Variable(tf.truncated_normal([num_units, int(Y.get_shape()[1])]),name="W")

    with tf.name_scope("biases"):
        B = tf.Variable(tf.constant(0.1, shape=[Y.get_shape()[1]]),name="B")

    # model-defination
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
    weights, _ = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
    weights = tf.transpose(weights, [1, 0, 2])
    last_weight = tf.gather(weights, int(weights.get_shape()[0]) - 1) # get the last weight

    with tf.name_scope("softmax"):
        pred = tf.nn.softmax(tf.matmul(last_weight, W) + B)

    with tf.name_scope('cross_entropy'):
        cost = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(cost)

    with tf.name_scope('Accuracy'):
        mistakes = tf.not_equal(tf.argmax(Y, 1), tf.argmax(pred, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    # create a summary for our cost and accuracy
    tf.scalar_summary("cost", cost)
    tf.scalar_summary("accuracy", error)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.merge_all_summaries()

    # train Parameters
    batch_size = 1000
    number_of_batch = train_limit//batch_size
    training_epoch = 500
    verbos_step=2
    saved_itr=100

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

        for i in range(training_epoch):
            ptr = 0
            for j in range(number_of_batch):
                inp, out = x_train[ptr:min(xlen,ptr + batch_size)], y_train[ptr:min(xlen,ptr + batch_size)]
                ptr += batch_size
                _,summary=sess.run([train_op,summary_op], {X: inp, Y: out})

                # write log
                writer.add_summary(summary, i * number_of_batch + j)

            if i % verbos_step == 0:
                incorrect = sess.run(error, {X: x_test, Y: y_test})
                print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

            if i% saved_itr==0:
                saver.save(sess,'model/setbit_count_model.ckpt',global_step=training_epoch)

        saver.save(sess,'model/setbit_count_model_final.ckpt',global_step=training_epoch)
        print "----> Train Completed <------"
        incorrect = sess.run(error, {X: x_test, Y: y_test})
        print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

def main():
    X,Y = generate_data(12)
    my_lstm(X,Y)

if __name__ == "__main__":
    main()
    # for tensorboard run on terminal
    # tensorboard --logdir=/home/afjal/GitHub/WorkingHouse/PycharmPro/AlphabetLearn/tensorflow/model/lstm/SetBitCount-LSTM/logs
