# coding=utf-8

# reference
# GRU description in http://colah.github.io/posts/2015-08-Understanding-LSTMs/

import tensorflow as tf
import numpy as np
from tensorflow.python import array_ops

from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    n_input = 28
    n_step = 28
    n_class = 10
    n_hidden = 4 * n_input
    batch_size = 100
    learning_rate = 1e-3
    n_iteration = 1e6
    valid_steps = 5000

    x = tf.placeholder(tf.float32, [None, n_step, n_input])
    y = tf.placeholder(tf.float32, [None, n_class])
    W_z = tf.Variable(tf.random_normal([n_input + n_hidden, n_hidden]))
    W_r = tf.Variable(tf.random_normal([n_input + n_hidden, n_hidden]))
    W_c = tf.Variable(tf.random_normal([n_input + n_hidden, n_hidden]))
    W_out = tf.Variable(tf.random_normal([n_hidden, n_class]))
    b_out = tf.Variable(np.zeros((n_class), np.float32))

    states = []
    for t in range(n_step):
        x_t = x[:, t, :]
        if t == 0:
            h_prev = tf.zeros(shape=[tf.shape(x)[0], n_hidden], dtype=tf.float32)
        else:
            h_prev = states[-1]
        hx = array_ops.concat(1, [h_prev, x_t])
        z_t = tf.sigmoid(tf.matmul(hx, W_z))
        r_t = tf.sigmoid(tf.matmul(hx, W_r))
        # h_c: hidden state candidate
        h_c = tf.tanh(tf.matmul(array_ops.concat(1, [r_t * h_prev, x_t]), W_c))
        h_t = (1 - z_t) * h_prev + z_t * h_c
        states.append(h_t)
    proba = tf.nn.softmax(tf.matmul(states[-1], W_out) + b_out)
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

    # with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    n_sample = train_x.shape[0]
    for i in range(int(n_iteration)):
        # validate the model
        if i % valid_steps == 0:
            loss, accu = sess.run([cost, accuracy], feed_dict={x: valid_x, y: valid_y})
            print '{i} batches fed in, valid set, loss {l:.4f}, accuracy {a:.2f}%'.format(i=i, l=loss, a=accu * 100.)
        # get a batch of samples
        idx = np.random.randint(0, n_sample, batch_size)
        batch_x = train_x[idx, :, :]
        batch_y = train_y[idx, :]
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
    # test the model
    accu = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print 'test set, accuracy {a:.2f}%'.format(a=accu * 100.)
    
