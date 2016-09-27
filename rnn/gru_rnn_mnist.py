# coding=utf-8

# reference
# GRU description in http://colah.github.io/posts/2015-08-Understanding-LSTMs/

import tensorflow as tf
import numpy as np
from tensorflow.python import array_ops
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
    def __init__(self, n_input, n_hidden):
        # initialise weights as normal distr
        self.W_z = weight_variable_normal([n_input + n_hidden, n_hidden])
        self.W_r = weight_variable_normal([n_input + n_hidden, n_hidden])
        self.W_c = weight_variable_normal([n_input + n_hidden, n_hidden])

    def __call__(self, h, x):
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
    
    n_input = 28
    n_step = 28
    n_class = 10
    n_hidden = [2*n_input, n_input]
    batch_size = 100
    learning_rate = 1e-2
    n_iteration = 1000
    validate_steps = 100
    display_steps = 100

    x = tf.placeholder(tf.float32, [None, n_step, n_input])
    y = tf.placeholder(tf.float32, [None, n_class])
    
    # initialise weights as normal distr
    W_out = weight_variable_normal([n_hidden[-1], n_class])
    b_out = tf.Variable(np.zeros((n_class), np.float32))

    # multi-layer GRUCells
    layer_size = [n_input] + n_hidden
    gru_cells = []
    for i in range(len(layer_size)-1):
        gru = GRUCell(layer_size[i], layer_size[i+1])
        gru_cells.append(gru)
        
    states = []
    for t in range(n_step):
        x_t = x[:, t, :]
        state = []
        for gru, i in zip(gru_cells, range(len(gru_cells))):
            if t == 0:
                h_prev = tf.zeros(shape=[tf.shape(x_t)[0], n_hidden[i]], dtype=tf.float32)
            else:
                h_prev = states[-1][i]
            h_t = gru(h_prev, x_t)
            state.append(h_t)
            x_t = h_t
        states.append(state)
    proba = tf.nn.softmax(tf.matmul(states[-1][-1], W_out) + b_out)
    pred = tf.argmax(proba, dimension=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(y, dimension=1)), tf.float32))
    cost = tf.reduce_mean(tf.reduce_sum(-y * tf.log(proba), 1))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_x = mnist.train.images.reshape([-1, 28, 28])
    train_y = mnist.train.labels
    valid_x = mnist.validation.images.reshape([-1, 28, 28])
    valid_y = mnist.validation.labels
    test_x = mnist.test.images.reshape([-1, 28, 28])
    test_y = mnist.test.labels

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    n_sample = train_x.shape[0]
    for i in range(int(n_iteration)):
        # validate the model
        if i % validate_steps == 0:
            loss, accu = sess.run([cost, accuracy], feed_dict={x: valid_x, y: valid_y})
            print '{i} batches fed in, valid set, loss {l:.4f}, accuracy {a:.2f}%'.format(i=i, l=loss, a=accu * 100.)
            saver.save(sess, 'tmp/gru.ckpt', global_step=i)
        # get a batch of samples
        idx = np.random.randint(0, n_sample, batch_size)
        batch_x = train_x[idx, :, :]
        batch_y = train_y[idx, :]
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        if i % display_steps == 0:
            loss, accu = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print '{i} samples fed in, training minibatch, loss {l:.4f}, accuracy {a:.2f}%'.format(i=i*batch_size, l=loss, a=accu * 100.)
    saver.save(sess, 'tmp/gru-final.ckpt')
    # test the model
    accu = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print 'test set, accuracy {a:.2f}%'.format(a=accu * 100.)
    sess.close()
    
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, 'tmp/gru-final.ckpt')
    # test the model
    accu = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print 'test set, accuracy {a:.2f}%'.format(a=accu * 100.)
    sess.close()