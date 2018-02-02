# -*- coding: UTF-8 -*-
import sys

import numpy as np
import tensorflow as tf


def bilinear(inputs1, weights, inputs2, output_size=1, add_bias1=False, add_bias2=False):
    """
    Perform inputs1 x weights x inputs2, (n x b x d) * (d x r x d) * (n x b x d).T -> (n x b x r x b)
    where n is batch size, b is max sequence length, (d x r x d) is the shape of tensor weights, r is output size

    Adopted from Timothy Dozat https://github.com/tdozat, with some modifications

    :param inputs1:
    :param weights:
    :param inputs2:
    :param output_size:
    :param add_bias1:
    :param add_bias2:
    :return:
    """
    # Prepare the input
    if not isinstance(inputs1, (list, tuple)):
        inputs1 = [inputs1]
    n_dims1 = len(inputs1[0].get_shape().as_list())
    all_inputs1 = tf.concat(inputs1, n_dims1 - 1)
    inputs1_size = all_inputs1.get_shape().as_list()[-1]
    inputs1_bucket_size = tf.shape(all_inputs1)[-2]

    if not isinstance(inputs2, (list, tuple)):
        inputs2 = [inputs2]
    n_dims2 = len(inputs2[0].get_shape().as_list())
    all_inputs2 = tf.concat(inputs2, n_dims2 - 1)
    inputs2_size = all_inputs2.get_shape().as_list()[-1]
    inputs2_bucket_size = tf.shape(all_inputs2)[-2]

    # Prepare the output
    output_shape = []
    shape1 = tf.shape(all_inputs1)
    for i in range(n_dims1 - 1):
        output_shape.append(shape1[i])
    output_shape.append(output_size)
    output_shape.append(inputs2_bucket_size)
    output_shape = tf.stack(output_shape)

    all_inputs1 = tf.reshape(all_inputs1, tf.stack([-1, inputs1_bucket_size, inputs1_size]))
    if add_bias1:
        bias1 = tf.ones(tf.stack([tf.shape(all_inputs1)[0], inputs1_bucket_size, 1]))
        all_inputs1 = tf.concat([all_inputs1, bias1], 2)
        inputs1_size += 1
    all_inputs2 = tf.reshape(all_inputs2, tf.stack([-1, inputs2_bucket_size, inputs2_size]))
    if add_bias2:
        bias2 = tf.ones(tf.stack([tf.shape(all_inputs2)[0], inputs2_bucket_size, 1]))
        all_inputs2 = tf.concat([all_inputs2, bias2], 2)
        inputs2_size += 1
    with tf.variable_scope('Bilinear'):
        tf.add_to_collection('Weights', weights)

        # Do the multiplication
        # (bn x d) (d x rd) -> (bn x rd)
        lin = tf.matmul(tf.reshape(all_inputs1, [-1, inputs1_size]),
                        tf.reshape(weights, [inputs1_size, -1]))
        # (b x nr x d) (b x n x d)T -> (b x nr x n)
        bilin = tf.matmul(tf.reshape(lin, tf.stack([-1, inputs1_bucket_size * output_size, inputs2_size])),
                          all_inputs2, transpose_b=True)
        # (bn x r x n)
        bilin = tf.reshape(bilin, output_shape)

        return bilin


def orthonormal_initializer(input_size, output_size, debug=False):
    """
    See https://hjweide.github.io/orthogonal-initialization-in-convolutional-layers

    Adopted from Timothy Dozat https://github.com/tdozat, with some modifications

    :param input_size:
    :param output_size:
    :param debug:
    :return:
    """
    if debug:
        # print((input_size, output_size))
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        return Q.astype(np.float32)

    if not tf.get_variable_scope().reuse:
        # print(tf.get_variable_scope().name)
        I = np.eye(output_size)
        lr = .1
        eps = .05 / (output_size + input_size)
        success = False
        while not success:
            Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
            for i in range(100):
                QTQmI = Q.T.dot(Q) - I
                loss = np.sum(QTQmI ** 2 / 2)
                Q2 = Q ** 2
                Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
                if np.isnan(Q[0, 0]):
                    lr /= 2
                    break
            if np.isfinite(loss) and np.max(Q) < 1e6:
                success = True
            eps *= 2
            # print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return Q.astype(np.float32)


def eprint(*args, **kwargs):
    """
    Print errors to stderr

    :param args:
    :param kwargs:
    """
    print(*args, file=sys.stderr, **kwargs)
