#!/usr/bin/env  python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# NA to np.nan
def converter(x):
    if x == 'NA':
        return np.nan
    else:
        return float(x)

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.33), name="weights")

def init_bias(shape):
    return tf.Variable(tf.zeros(shape), name="biases")

# 推論
def inference(x, w, b):
    lm = tf.matmul(x, w) + b
    return lm

# 損失関数
def loss(lm, y):
    return tf.reduce_mean(tf.square(lm - y))

# 学習
def training(loss, rate):
    return tf.train.AdagradOptimizer(rate).minimize(loss)

def main():
    data = np.loadtxt("data/airquality.csv", delimiter=",",
                      skiprows=1, converters={0: converter, 1: converter})
    data = data[~np.isnan(data).any(axis=1)]
    #   Ozone Solar.R Wind Temp Month Day
    # 1    41     190  7.4   67     5   1
    # 2    36     118  8.0   72     5   2
    # 3    12     149 12.6   74     5   3
    # 4    18     313 11.5   62     5   4
    # 7    23     299  8.6   65     5   7
    # 8    19      99 13.8   59     5   8

    N = data.shape[0]
    train_x = data[:, 1:].astype(np.float32)
    train_y = data[:, :1].astype(np.float32).reshape(N, 1)

    # symbolic variables
    x = tf.placeholder(tf.float32, shape=(N, 5))
    y = tf.placeholder(tf.float32, shape=(N, 1))

    # parameters
    w = init_weight([5, 1])
    b = init_bias([1])

    model = inference(x, w, b)
    loss_value = loss(model, y)
    train_op = training(loss_value, 1.)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in xrange(30001):
            sess.run(train_op, feed_dict={x: train_x, y: train_y})
            if i % 1000 == 0:
                cost = sess.run(loss_value, feed_dict={x: train_x, y: train_y})
                pred_y = sess.run(model, feed_dict={x: train_x, y: train_y})
                cor = np.corrcoef(pred_y.flatten(), train_y.flatten())
                print('step : {}, loss : {}, cor : {}, w : {}, b: {}'.format(
                    i, cost, cor[0][1], sess.run(w), sess.run(b)))

if __name__ == '__main__':
    main()
