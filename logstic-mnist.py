#!/usr/bin/env  python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 推論
def inference(x, w):
    return tf.matmul(x, w)

# 損失関数
def loss(model, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))

# 学習
def training(loss, rate):
    return tf.train.GradientDescentOptimizer(rate).minimize(loss)

if __name__ == '__main__':
    mnist = input_data.read_data_sets("data/", one_hot=True)
    train_x, train_y = mnist.train.images, mnist.train.labels
    test_x, test_y  = mnist.test.images, mnist.test.labels

    # symbolic variables
    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None, 10])

    # parameters
    w = init_weights([784, 10])

    model = inference(x, w)
    loss_value = loss(model, y)
    train_op = training(loss_value, 0.03)
    predict_op = tf.argmax(model, 1)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(10):
            for start, end in zip(range(0, len(train_x), 128), range(128, len(train_x), 128)):
                sess.run(train_op, feed_dict={x: train_x[start:end], y: train_y[start:end]})
            prob = np.mean(np.argmax(test_y, axis=1) ==
                           sess.run(predict_op, feed_dict={x: test_x, y: test_y}))
            print('step : {}, test prob : {}'.format(step, prob))

        print'weights : {}'.format(sess.run(w))
