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


if __name__ == '__main__':
    n_input = 28
    n_step = 28
    n_class = 10
    n_hidden = 4 * n_input
    batch_size = 100
    learning_rate = 1e-2
    n_iteration = 200
    valid_steps = 100

    x = tf.placeholder(tf.float32, [None, n_step, n_input])
    y = tf.placeholder(tf.float32, [None, n_class])
    
    # initialise weights as normal distr
    W_z = weight_variable_normal([n_input + n_hidden, n_hidden])
    W_r = weight_variable_normal([n_input + n_hidden, n_hidden])
    W_c = weight_variable_normal([n_input + n_hidden, n_hidden])
    W_out = weight_variable_normal([n_hidden, n_class])
    b_out = tf.Variable(np.zeros((n_class), np.float32))

    ## initialise weights as uniform distr
    #W_z = weight_variable_uniform(shape=[n_input + n_hidden, n_hidden], radius=0.05)
    #W_r = weight_variable_uniform(shape=[n_input + n_hidden, n_hidden], radius=0.05)
    #W_c = weight_variable_uniform(shape=[n_input + n_hidden, n_hidden], radius=0.05)
    #W_out = weight_variable_uniform(shape=[n_hidden, n_class], radius=0.05)
    #b_out = tf.Variable(np.zeros((n_class), np.float32))

    states = []
    for t in range(n_step):
        x_t = x[:, t, :]
        if t == 0:
            h_prev = tf.zeros(shape=[tf.shape(x)[0], n_hidden], dtype=tf.float32)
        else:
            h_prev = states[-1]
        hx = array_ops.concat(1, [h_prev, x_t])
        # z: update gate
        z_t = tf.sigmoid(tf.matmul(hx, W_z))
        # r: reset gate
        r_t = tf.sigmoid(tf.matmul(hx, W_r))
        # h_c: candidate hidden state
        h_c = tf.tanh(tf.matmul(array_ops.concat(1, [r_t * h_prev, x_t]), W_c))
        h_t = (1 - z_t) * h_prev + z_t * h_c
        states.append(h_t)
    proba = tf.nn.softmax(tf.matmul(states[-1], W_out) + b_out)
    pred = tf.argmax(proba, dimension=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(y, dimension=1)), tf.float32))
    cost = tf.reduce_mean(tf.reduce_sum(-y * tf.log(proba), 1))
    
    ## use the wrapped Adam
    #train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # calc gradients myself
    tvars = [W_z, W_r, W_c, W_out, b_out]
    grads = tf.gradients(cost, tvars)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = opt.apply_gradients(zip(grads, tvars))
    grads_mag = tf.pack([tf.reduce_mean(tf.abs(g)) for g in grads])
    tavr_mag = tf.pack([tf.reduce_mean(tf.abs(v)) for v in tvars])
    
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
        #if i % valid_steps == 0:
        #    loss, accu = sess.run([cost, accuracy], feed_dict={x: valid_x, y: valid_y})
        #    print '{i} batches fed in, valid set, loss {l:.4f}, accuracy {a:.2f}%'.format(i=i, l=loss, a=accu * 100.)
        # get a batch of samples
        idx = np.random.randint(0, n_sample, batch_size)
        batch_x = train_x[idx, :, :]
        batch_y = train_y[idx, :]
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        if i % 10 == 0:
            loss, accu, gradient_mag, weights_mag = sess.run([cost, accuracy, grads_mag, tavr_mag], feed_dict={x: batch_x, y: batch_y})
            print '{i} samples fed in, training minibatch, loss {l:.4f}, accuracy {a:.2f}%'.format(i=i*batch_size, l=loss, a=accu * 100.)
            print '\tlog(avg grad) {g:.3f}, avg weight {w:.3f}'.format(g=np.log10(np.mean(gradient_mag)), w=np.mean(weights_mag))
    # test the model
    accu = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print 'test set, accuracy {a:.2f}%'.format(a=accu * 100.)
    sess.close()
    
