# -*- coding: utf-8 -*-

import tensorflow as tf
import multiprocessing as mp


def main():
    core_num = mp.cpu_count()
    config = tf.ConfigProto(
                inter_op_parallelism_threads=core_num,
                intra_op_parallelism_threads=core_num
             )
    sess = tf.Session(config=config)

    hello = tf.constant('hello, tensorflow!')
    print(sess.run(hello))

    a = tf.constant(10)
    b = tf.constant(32)
    print(sess.run(a + b))


if __name__ == '__main__':
    main()
