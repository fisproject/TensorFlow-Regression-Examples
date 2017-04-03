#!/usr/bin/env  python2.7
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn import datasets

# 推論
def inference(x, p_keep_in, p_keep_hidden):
    n = [128, 256, 1]

    with tf.name_scope('l1') as scope:
        w_l1 = tf.Variable(tf.truncated_normal(
            [10, n[0]], stddev=0.33), name="w_l1")
        b_l1 = tf.Variable(tf.constant(
            1.0, shape=[n[0]]), name="b_l1")
        h_l1 = tf.nn.relu(tf.matmul(x, w_l1) + b_l1)
        h_l1 = tf.nn.dropout(h_l1, p_keep_in)

    with tf.name_scope('l2') as scope:
        w_l2 = tf.Variable(tf.truncated_normal(
            [n[0], n[1]], stddev=0.33), name="w_l2")
        b_l2 = tf.Variable(tf.constant(
            1.0, shape=[n[1]]), name="b_l2")
        h_l2 = tf.nn.relu(tf.matmul(h_l1, w_l2) + b_l2)
        h_l2 = tf.nn.dropout(h_l2, p_keep_in)

    with tf.name_scope('l3') as scope:
        w_l3 = tf.Variable(tf.truncated_normal(
            [n[1], n[2]], stddev=0.33), name="w_l3")
        b_l3 = tf.Variable(tf.constant(
            1.0, shape=[n[2]]), name="b_l3")
        output = tf.nn.relu(tf.matmul(h_l2, w_l3) + b_l3)
        output = tf.nn.dropout(output, p_keep_hidden)

    return output

# 損失関数
def loss(model, y):
    return tf.reduce_mean(tf.square(model - y), name="loss")

# 学習
def training(loss, rate):
    return tf.train.RMSPropOptimizer(rate, 0.9).minimize(loss)

if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    data = diabetes["data"].astype(np.float32)
    target = diabetes['target'].astype(
        np.float32).reshape(len(diabetes['target']), 1)

    MAX_SIZE = data.shape[0]
    TEST_N = 100
    N = MAX_SIZE - TEST_N
    BATCH_SIZE = 10
    MAX_STEPS = 300

    train_x, test_x = np.vsplit(data, [N])
    train_y, test_y = np.vsplit(target, [N])

    # symbolic variables
    x = tf.placeholder(tf.float32, shape=[None, 10])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    p_keep_in = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)

    model = inference(x, p_keep_in, p_keep_hidden)
    loss_value = loss(model, y)
    train_op = training(loss_value, 0.001)

    best = float("inf")
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for step in xrange(MAX_STEPS + 1):
            for i in xrange(N / BATCH_SIZE):
                batch = BATCH_SIZE * i
                train_batch_x = train_x[batch:batch + BATCH_SIZE]
                train_batch_y = train_y[batch:batch + BATCH_SIZE]

                loss_train = sess.run(loss_value, feed_dict={
                                      x: train_batch_x, y: train_batch_y,
                                      p_keep_in: 1.0, p_keep_hidden: 1.0})
                sess.run(train_op, feed_dict={
                         x: train_batch_x, y: train_batch_y,
                         p_keep_in: 0.8, p_keep_hidden: 0.5})

            if loss_train < best:
                best = loss_train
                best_match = sess.run(model, feed_dict={
                    x: test_x, y: test_y, p_keep_in: 1.0, p_keep_hidden: 1.0})

            if step % 10 == 0:
                cor = np.corrcoef(best_match.flatten(), test_y.flatten())
                print('step : {}, train loss : {}, test cor : {}'.format(
                    step, best, cor[0][1]))
