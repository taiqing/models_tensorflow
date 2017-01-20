# coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable_normal(shape, stddev=None):
    if stddev is not None:
        std = stddev
    else:
        std = 1.0 / np.sqrt(shape[0])
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=std)
    return tf.Variable(initial)


def weight_variable_uniform(shape, radius):
    initial = tf.random_uniform(shape=shape, minval=-radius, maxval=radius)
    return tf.Variable(initial)


class GRUCell(object):
    # GRU description in http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    
    def __init__(self, n_input, n_hidden):
        # initialise weights as normal distr
        self.W_z = weight_variable_normal([n_input + n_hidden, n_hidden])
        self.W_r = weight_variable_normal([n_input + n_hidden, n_hidden])
        self.W_c = weight_variable_normal([n_input + n_hidden, n_hidden])
        self.paramters = [self.W_z, self.W_r, self.W_c]

    def __call__(self, h, x):
        """
        :param h: must be rank-2
        :param x: must be rank-2
        :return:
        """
        hx = array_ops.concat(1, [h, x])
        # z: update gate
        z = tf.sigmoid(tf.matmul(hx, self.W_z))
        # r: reset gate
        r = tf.sigmoid(tf.matmul(hx, self.W_r))
        # h_c: candidate hidden state
        h_candidate = tf.tanh(tf.matmul(array_ops.concat(1, [r * h, x]), self.W_c))
        new_h = (1 - z) * h + z * h_candidate
        return new_h


if __name__ == '__main__':
    tf.reset_default_graph()
    
    n_input = 14
    n_output = n_input
    n_step = 14
    n_hidden = 2 * n_input
    gamma = 1e-3
    learning_rate = 1e-2
    n_epochs = 1
    validation_steps = 5000

    x = tf.placeholder(tf.float32, [n_step, n_input])

    # encoder
    h0 = tf.zeros((1, n_hidden), tf.float32)
    encoder_cell = GRUCell(n_input, n_hidden)
    encoder_states = [h0]
    for i in range(n_step-1, -1, -1):
        h_prev = encoder_states[-1]
        x_t = tf.reshape(x[i, :], [1, -1])
        h_new = encoder_cell(h_prev, x_t) # reads input in reverse order
        encoder_states.append(h_new)

    # decoder
    decoder_cell = GRUCell(n_input, n_hidden)
    decoder_states = [encoder_states[-1]]
    initial_input = tf.zeros([1, n_input], tf.float32)
    for t in range(0, n_step):
        h_prev = decoder_states[-1]
        x_t = initial_input if t == 0 else tf.reshape(x[t - 1, :], [1, -1])
        h_new = decoder_cell(h_prev, x_t)
        decoder_states.append(h_new)

    # output
    W_o = weight_variable_normal([n_hidden, n_output])
    b_o = tf.Variable(np.zeros(n_output, dtype=np.float32))
    outputs = list()
    for i in range(1, len(decoder_states)):
        h = decoder_states[i]
        out = tf.sigmoid(tf.matmul(h, W_o) + b_o)
        outputs.append(out)
    outputs = tf.concat(0, outputs) # outputs: n_step x n_output    
    
    # loss
    loss = tf.reduce_mean(tf.squared_difference(outputs, x))

    # l2-norm of paramters
    regularizer = 0.
    for cell in [encoder_cell, decoder_cell]:
        for param in cell.paramters:
            regularizer += tf.reduce_mean(tf.square(param))

    # cost
    cost = loss + gamma * regularizer
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_x = mnist.train.images.reshape([-1, 28, 28])[:, ::2, ::2]
    print '{} training samples'.format(train_x.shape[0])
    validation_x = mnist.validation.images.reshape([-1, 28, 28])[:, ::2, ::2]
    print '{} validation samples'.format(validation_x.shape[0])
    test_x = mnist.test.images.reshape([-1, 28, 28])[:, ::2, ::2]
    print '{} test samples'.format(test_x.shape[0])

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    n_sample = train_x.shape[0]
    np.random.seed(1001)
    for i in range(int(n_epochs * n_sample)):
        idx = np.random.randint(n_sample)
        sample = train_x[idx, :, :]
        sess.run(train_step, feed_dict={x: sample})
        if i % int(validation_steps) == 0:
            validation_cost = 0.
            validation_loss = 0.
            parameter_regu = 0.
            for j in range(validation_x.shape[0]):
                c, l, r = sess.run([cost, loss, regularizer], feed_dict={x: validation_x[j, :, :]})
                validation_cost += c
                validation_loss += l
                parameter_regu += r
            validation_cost /= validation_x.shape[0]
            validation_loss /= validation_x.shape[0]
            parameter_regu /= validation_x.shape[0]
            print 'iteration {i}: validation: {n} samples, cost {c}, loss {l}, paramter regularizer {r}'.format(
                i=i,
                n=validation_x.shape[0],
                c=validation_cost,
                l=validation_loss,
                r=parameter_regu
            )
    saver.save(sess, 'seq_to_seq_rnn-final.ckpt')
    sess.close()