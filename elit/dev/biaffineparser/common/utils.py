# -*- coding: UTF-8 -*-
import sys

import numpy as np
import tensorflow as tf


def bilinear(inputs1, inputs2, output_size, n_splits=1, add_bias1=True, add_bias2=True, initializer=None):
    """
    Perform inputs1 x weights x inputs2, (n x b x d) * (d x r x d) * (n x b x d).T -> (n x b x r x b)
    where n is batch size, b is max sequence length, (d x r x d) is the shape of tensor weights, r is output size

        Adopted from Timothy Dozat https://github.com/tdozat, with some modifications

    :param inputs1:
    :param inputs2:
    :param output_size:
    :param n_splits:
    :param add_bias1:
    :param add_bias2:
    :param initializer:
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
    output_size *= n_splits
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
        # Get the matrix
        if initializer is None and tf.get_variable_scope().reuse is None:
            mat = orthonormal_initializer(inputs1_size, inputs2_size)[:, None, :]
            mat = np.concatenate([mat] * output_size, axis=1)
        weights = tf.get_variable('Weights', [inputs1_size, output_size, inputs2_size], initializer=initializer)
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


def gate(x):
    return tf.nn.sigmoid(2 * x)


def tanh(x):
    return tf.nn.tanh(x)


def leaky_relu(x):
    return tf.maximum(.1 * x, x)


def linear(inputs, output_size, n_splits=1, add_bias=True, initializer=None):
    """
    y = Wx + b
    :param inputs: x
    :param output_size: dim of y
    :param n_splits: How many MLPs are there? [y_1...y_n] = W[x_1...x_n] + b
    :param add_bias: if false then b = 0
    :param initializer: initializer for W
    :return: y
        Adopted from Timothy Dozat https://github.com/tdozat, with some modifications
    """

    # Prepare the input
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    n_dims = len(inputs[0].get_shape().as_list())
    all_inputs = tf.concat(inputs, n_dims - 1)
    input_size = all_inputs.get_shape().as_list()[-1]

    # Prepare the output
    output_size *= n_splits
    output_shape = []
    shape = tf.shape(all_inputs)
    for i in range(n_dims - 1):
        output_shape.append(shape[i])
    output_shape.append(output_size)
    output_shape = tf.stack(output_shape)

    all_inputs = tf.reshape(all_inputs, [-1, input_size])
    with tf.variable_scope('Linear'):
        # Get the matrix
        if initializer is None and not tf.get_variable_scope().reuse:
            mat = orthonormal_initializer(input_size, output_size // n_splits)
            mat = np.concatenate([mat] * n_splits, axis=1)
            initializer = tf.constant_initializer(mat)
        matrix = tf.get_variable('Weights', [input_size, output_size], initializer=initializer)
        # if moving_params is not None:
        #     matrix = moving_params.average(matrix)
        # else:
        #     tf.add_to_collection('Weights', matrix)
        tf.add_to_collection('Weights', matrix)

        # Get the bias
        if add_bias:
            bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
            # if moving_params is not None:
            #     bias = moving_params.average(bias)

        else:
            bias = 0

        # Do the multiplication
        lin = tf.matmul(all_inputs, matrix) + bias
        lin = tf.reshape(lin, output_shape)
        if n_splits > 1:
            return tf.split(lin, n_splits, n_dims - 1)
        else:
            return lin


def birnn(cell, inputs, sequence_length, initial_state_fw=None, initial_state_bw=None, ff_keep_prob=1.,
          recur_keep_prob=1., dtype=tf.float32, scope=None):
    """
    Bi-RNN. Define b as batch, t as time step, d as dimension of feature
        Adopted from Timothy Dozat https://github.com/tdozat, with some modifications
    :param cell: RNN cell
    :param inputs: (b, n, d)
    :param sequence_length: (b,)
    :param initial_state_fw: (d,)
    :param initial_state_bw: (d,)
    :param ff_keep_prob: (,)
    :param recur_keep_prob: (,)
    :param dtype:
    :param scope:
    :return:
    """

    # Forward direction
    with tf.variable_scope(scope or 'BiRNN_FW') as fw_scope:
        output_fw, output_state_fw = rnn(cell, inputs, sequence_length, initial_state_fw, ff_keep_prob, recur_keep_prob,
                                         dtype, scope=fw_scope)

    # Backward direction
    rev_inputs = tf.reverse_sequence(inputs, sequence_length, 1, 0)
    with tf.variable_scope(scope or 'BiRNN_BW') as bw_scope:
        output_bw, output_state_bw = rnn(cell, rev_inputs, sequence_length, initial_state_bw, ff_keep_prob,
                                         recur_keep_prob, dtype, scope=bw_scope)
    output_bw = tf.reverse_sequence(output_bw, sequence_length, 1, 0)
    # Concat each of the forward/backward outputs
    outputs = tf.concat([output_fw, output_bw], 2)

    return outputs, tf.tuple([output_state_fw, output_state_bw])


def rnn(cell, inputs, sequence_length=None, initial_state=None, ff_keep_prob=1., recur_keep_prob=1., dtype=tf.float32,
        scope=None):
    """
    RNN. Similar to birnn
        Adopted from Timothy Dozat https://github.com/tdozat, with some modifications
    :param cell:
    :param inputs:
    :param sequence_length:
    :param initial_state:
    :param ff_keep_prob:
    :param recur_keep_prob:
    :param dtype:
    :param scope:
    :return:
    """

    inputs = tf.transpose(inputs, [1, 0, 2])  # (B,T,D) => (T,B,D)

    parallel_iterations = 32
    if sequence_length is not None:
        sequence_length = tf.to_int32(sequence_length)

    with tf.variable_scope(scope or 'RNN') as varscope:
        # if varscope.caching_device is None:
        #  varscope.set_caching_device(lambda op: op.device)
        input_shape = tf.shape(inputs)
        time_steps, batch_size, _ = tf.unstack(input_shape, 3)
        const_time_steps, const_batch_size, const_depth = inputs.get_shape().as_list()

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError('If no initial_state is provided, dtype must be.')
            state = cell.zero_state(batch_size, dtype)

        zero_output = tf.zeros(tf.stack([batch_size, cell.output_size]), inputs.dtype)
        if sequence_length is not None:
            min_sequence_length = tf.reduce_min(sequence_length)
            max_sequence_length = tf.reduce_max(sequence_length)

        time = tf.constant(0, dtype=tf.int32, name='time')

        output_ta = tf.TensorArray(dtype=inputs.dtype,
                                   size=time_steps,
                                   tensor_array_name='dynamic_rnn_output')

        input_ta = tf.TensorArray(dtype=inputs.dtype,
                                  size=time_steps,
                                  tensor_array_name='dynamic_rnn_input')

        def dropout_inputs(inputs):
            noise_shape = tf.stack([1, batch_size, const_depth])
            inputs = tf.nn.dropout(inputs, ff_keep_prob, noise_shape=noise_shape)
            return inputs

        def dropout_mask_state():
            ones = tf.ones(tf.stack([batch_size, cell.output_size]))
            state_dropout = tf.nn.dropout(ones, recur_keep_prob)
            state_dropout = tf.concat([ones] * (cell.state_size // cell.output_size - 1) + [state_dropout], 1)
            return state_dropout

        inputs = tf.cond(ff_keep_prob < 1, lambda: dropout_inputs(inputs), lambda: inputs)
        state_dropout = tf.cond(recur_keep_prob < 1, lambda: dropout_mask_state(),
                                lambda: tf.ones(
                                    tf.stack([batch_size, cell.output_size * cell.state_size // cell.output_size])))

        input_ta = input_ta.unstack(inputs)

        # -----------------------------------------------------------
        def _time_step(time, state, output_ta_t):
            """"""

            input_t = input_ta.read(time)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            def _empty_update():
                return zero_output, state

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            def _call_cell():
                return cell(input_t, state * state_dropout)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            def _maybe_copy_some_through():
                new_output, new_state = _call_cell()

                return tf.cond(
                    time < min_sequence_length,
                    lambda: (new_output, new_state),
                    lambda: (tf.where(time >= sequence_length, zero_output, new_output),
                             tf.where(time >= sequence_length, state, new_state)))

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            if sequence_length is not None:
                output, new_state = tf.cond(
                    time >= max_sequence_length,
                    _empty_update,
                    _maybe_copy_some_through)
            else:
                (output, new_state) = _call_cell()

            output_ta_t = output_ta_t.write(time, output)

            return (time + 1, new_state, output_ta_t)

        # -----------------------------------------------------------

        _, final_state, output_final_ta = tf.while_loop(
            cond=lambda time, _1, _2: time < time_steps,
            body=_time_step,
            loop_vars=(time, state, output_ta),
            parallel_iterations=parallel_iterations)

        final_outputs = output_final_ta.stack()

        outputs = tf.transpose(final_outputs, [1, 0, 2])  # (T,B,D) => (B,T,D)
        return outputs, final_state
