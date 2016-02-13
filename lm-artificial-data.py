#!/usr/bin/env  python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# 推論
def inference(w, b):
    lm = x * w + b
    return lm

# 損失関数
def loss(lm, y):
    return tf.reduce_mean(tf.square(lm - y))

# 学習
def training(loss, rate):
    return tf.train.AdagradOptimizer(rate).minimize(loss)

if __name__ == '__main__':
    N = 101

    train_x = np.linspace(-1, 1, N)
    err = np.random.randn(*train_x.shape) * 0.5
    train_y = 2 * train_x + 3 + err

    # parameters
    w = tf.Variable([0.])
    b = tf.Variable([0.])

    # symbolic variables
    x = tf.placeholder(tf.float32, shape=(N))
    y = tf.placeholder(tf.float32, shape=(N))

    model = inference(w, b)
    loss_value = loss(model, y)
    train_op = training(loss_value, 1.)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1001):
            sess.run(train_op, feed_dict={x: train_x, y: train_y})
            if i % 100 == 0:
                print('step : {}, w : {}, b: {}'.format(
                    i, sess.run(w), sess.run(b)))
