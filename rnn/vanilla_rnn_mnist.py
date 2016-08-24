# coding=utf-8

import tensorflow as tf
import numpy as np

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


if __name__ == '__main__':
    n_input = 28
    n_step = 28
    n_class = 10
    n_hidden = 4 * n_input
    batch_size = 100
    learning_rate = 1e-2
    n_iteration = 500
    valid_steps = 1e3

    x = tf.placeholder(tf.float32, [None, n_step, n_input])
    y = tf.placeholder(tf.float32, [None, n_class])
    U = weight_variable_normal([n_input, n_hidden])
    W = weight_variable_normal([n_hidden, n_hidden])
    b = tf.Variable(np.zeros((n_hidden), dtype=np.float32))
    V = weight_variable_normal([n_hidden, n_class])
    c = tf.Variable(np.zeros((n_class), dtype=np.float32))
    
    states = []
    for t in range(n_step):
        x_t = x[:, t, :]
        if len(states) < 1:
            s_t = tf.tanh(tf.matmul(x_t, U) + b)
        else:
            s_t = tf.tanh(tf.matmul(x_t, U) + tf.matmul(states[-1], W) + b)
        states.append(s_t)
    proba = tf.nn.softmax(tf.matmul(states[-1], V) + c)
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
    
    #with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    n_sample = train_x.shape[0]
    for i in range(int(n_iteration)):
        ## validate the model
        #if i % valid_steps == 0:
        #    loss, accu = sess.run([cost, accuracy], feed_dict={x: valid_x, y: valid_y})
        #    print '{i} batches fed in, valid set, loss {l:.4f}, accuracy {a:.2f}%'.format(i=i, l=loss, a=accu * 100.)
        # get a batch of samples
        idx = np.random.randint(0, n_sample, batch_size)
        batch_x = train_x[idx, :, :]
        batch_y = train_y[idx, :]
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        if i % 10 == 0:
            loss, accu, state = sess.run([cost, accuracy, states[-1]], feed_dict={x: batch_x, y: batch_y})
            print '{i} samples fed in, training minibatch, loss {l:.4f}, accuracy {a:.2f}%'.format(i=i*batch_size, l=loss, a=accu * 100.)
    # test the model
    accu = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print 'test set, accuracy {a:.2f}%'.format(a=accu * 100.)
    sess.close()