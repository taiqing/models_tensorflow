# coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle

from vis_util import tile_raster_images


def join_dicts(dict1, dict2):
    """
    Raise exception if two dicts share some keys
    """
    dict_ret = dict1.copy()
    for k, v in dict2.iteritems():
        if k not in dict_ret:
            dict_ret[k] = v
        else:
            raise Exception('Key conflicts in join_dicts')
    return dict_ret


def weight_variable_normal(shape, stddev=None):
    if stddev is None:
        stddev = 1.0 / np.sqrt(shape[0])
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=stddev)
    return tf.Variable(initial)


def weight_variable_uniform(shape, radius=None):
    if radius is None:
        radius = 1.0 / np.sqrt(shape[0])
    initial = tf.random_uniform(shape=shape, minval=-radius, maxval=radius)
    return tf.Variable(initial)


class GRUCell(object):
    # GRU description in http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    
    def __init__(self, n_input, n_hidden, stddev=None, variable_values=None, name='GRU'):
        if variable_values is None:
            # update gate
            self.W_z = weight_variable_normal([n_input + n_hidden, n_hidden], stddev)
            self.b_z = tf.Variable(tf.zeros(n_hidden, tf.float32))
            # reset gate
            self.W_r = weight_variable_normal([n_input + n_hidden, n_hidden], stddev)
            self.b_r = tf.Variable(tf.zeros(n_hidden, tf.float32))
            # candidate generation
            self.W_c = weight_variable_normal([n_input + n_hidden, n_hidden], stddev)
            self.b_c = tf.Variable(tf.zeros(n_hidden, tf.float32))
        else:
            self.W_z = tf.Variable(variable_values[':'.join([name, 'W_z'])])
            self.b_z = tf.Variable(variable_values[':'.join([name, 'b_z'])])
            self.W_r = tf.Variable(variable_values[':'.join([name, 'W_r'])])
            self.b_r = tf.Variable(variable_values[':'.join([name, 'b_r'])])
            self.W_c = tf.Variable(variable_values[':'.join([name, 'W_c'])])
            self.b_c = tf.Variable(variable_values[':'.join([name, 'b_c'])])
        
        self.vars = {':'.join([name, 'W_z']): self.W_z,
                     ':'.join([name, 'b_z']): self.b_z,
                     ':'.join([name, 'W_r']): self.W_r,
                     ':'.join([name, 'b_r']): self.b_r,
                     ':'.join([name, 'W_c']): self.W_c,
                     ':'.join([name, 'b_c']): self.b_c}

    def __call__(self, h, x):
        """
        :param h: must be rank-2
        :param x: must be rank-2
        :return:
        """
        hx = array_ops.concat(1, [h, x])
        # z: update gate
        z = tf.sigmoid(tf.matmul(hx, self.W_z) + self.b_z)
        # r: reset gate
        r = tf.sigmoid(tf.matmul(hx, self.W_r) + self.b_r)
        # h_c: candidate hidden state
        h_candidate = tf.tanh(tf.matmul(array_ops.concat(1, [r * h, x]), self.W_c) + self.b_c)
        new_h = (1 - z) * h + z * h_candidate
        return new_h


class TemporalAutoEncoder(object):
    """ one-layer temporal auto encoder """
    def __init__(self, n_input=None, n_step=None, n_hidden=None, stddev=None):
        self.n_input = n_input
        self.n_output = self.n_input
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.stddev = stddev

        # parameters to learn: values of the trainable variables
        self.parameters = dict()

        # for predicting
        self.sess_serve = None

    def __del__(self):
        if self.sess_serve is not None:
            self.sess_serve.close()

    def __build_graph__(self, variable_values=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            encoder_cell = GRUCell(self.n_input, self.n_hidden, self.stddev, variable_values=variable_values, name='encoder:0')
            decoder_cell = GRUCell(self.n_input, self.n_hidden, self.stddev, variable_values=variable_values, name='decoder:0')
            if variable_values is None:
                W_o = weight_variable_normal([self.n_hidden, self.n_output], self.stddev)
                b_o = tf.Variable(np.zeros(self.n_output, dtype=np.float32))
            else:
                W_o = tf.Variable(variable_values['W_o'])
                b_o = tf.Variable(variable_values['b_o'])

            # variables
            self.variables = join_dicts(join_dicts(encoder_cell.vars, decoder_cell.vars), {'W_o': W_o, 'b_o': b_o})

            # placeholders
            x = tf.placeholder(tf.float32, [None, self.n_step, self.n_input], name='x')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            gamma = tf.placeholder(tf.float32, name='gamma')

            # encoding
            n_sample = tf.shape(x)[0]
            h0 = tf.zeros((n_sample, self.n_hidden), tf.float32)
            encoder_states = [h0]
            for i in range(self.n_step-1, -1, -1):
                h_prev = encoder_states[-1]
                x_t = x[:, i, :]
                h_t = encoder_cell(h_prev, x_t)  # reads input in reverse order
                encoder_states.append(h_t)

            # decoding
            decoder_states = [encoder_states[-1]]
            initial_input = tf.zeros([n_sample, self.n_input], tf.float32)
            for t in range(0, self.n_step):
                h_prev = decoder_states[-1]
                x_t = initial_input if t == 0 else x[:, t - 1, :]
                h_t = decoder_cell(h_prev, x_t)
                decoder_states.append(h_t)

            # output
            outputs = list()
            for i in range(1, len(decoder_states)):
                h = decoder_states[i]
                out = tf.sigmoid(tf.matmul(h, W_o) + b_o)
                outputs.append(out)
            outputs = tf.pack(outputs, axis=1)  # outputs: n_samples x n_step x n_output

            # serving
            decoder_states_run = [encoder_states[-1]]
            outputs_run = list()
            initial_input = tf.zeros([n_sample, self.n_input], tf.float32)
            for t in range(0, n_step):
                h_prev = decoder_states_run[-1]
                x_t = initial_input if t == 0 else outputs_run[-1]
                h_t = decoder_cell(h_prev, x_t)
                out_t = tf.sigmoid(tf.matmul(h_t, W_o) + b_o)
                outputs_run.append(out_t)
                decoder_states_run.append(h_t)
            outputs_run = tf.pack(outputs_run, axis=1)  # outputs: n_samples x n_step x n_output

            # loss
            loss = tf.reduce_mean(tf.squared_difference(outputs, x))

            # l2-norm of paramters
            regularizer = 0.
            for k, v in self.variables.iteritems():
                regularizer += tf.reduce_mean(tf.square(v))

            # cost
            cost = loss + gamma * regularizer
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            init_vars = tf.global_variables_initializer()

        self.placeholders = dict(x=x, learning_rate=learning_rate, gamma=gamma)
        self.tensors = dict(cost=cost, loss=loss, regularizer=regularizer, outputs_run=outputs_run)
        self.operations = dict(train_step=train_step, init_vars=init_vars)

    def fit(self, train_x, validation_x, learning_rate, gamma, n_epoch, batch_size, validation_steps):
        """
        :param train_x: np.ndarray of size (n_sample, n_step, n_input)
        :param validation_x: np.ndarray of size (n_sample, n_step, n_input)
        :param learning_rate:
        :param gamma:
        :param n_epoch:
        :param validation_steps:
        :return:
        """
        np.random.seed(1001)
        n_sample = train_x.shape[0]

        self.__build_graph__()
        sess = tf.Session(graph=self.graph)
        with sess.as_default():
            self.operations['init_vars'].run()
            for i in range(int(n_epoch * n_sample / batch_size)):
                selected_idx = np.random.permutation(n_sample)[0:batch_size]
                x = train_x[selected_idx, :, :]
                self.operations['train_step'].run(feed_dict={
                    self.placeholders['x']: x,
                    self.placeholders['learning_rate']: learning_rate,
                    self.placeholders['gamma']: gamma})
                if i % int(validation_steps) == 0:
                    cost, loss, regu = sess.run([self.tensors['cost'], self.tensors['loss'], self.tensors['regularizer']],
                                           feed_dict={self.placeholders['x']: validation_x, self.placeholders['gamma']: gamma})
                    print 'iteration {i}: validation: {n} samples, cost {c:.5f}, loss {l:.5f}, paramter regularizer {r:.5f}'.format(
                        i=i,
                        n=validation_x.shape[0],
                        c=cost,
                        l=loss,
                        r=regu
                    )
            for k, v in self.variables.iteritems():
                self.parameters[k] = sess.run(v)

    def predict(self, x):
        """
        :param x: np.ndarray of size (n_sample, n_step, n_input)
        :return: np.ndarray of size (n_sample, n_step, n_output)
        """
        if self.sess_serve is None:
            self.sess_serve = tf.Session(graph=self.graph)
        feed_dict = dict()
        for k, v in self.variables.iteritems():
            feed_dict[v] = self.parameters[k]
        feed_dict[self.placeholders['x']] = x
        output = self.sess_serve.run(self.tensors['outputs_run'],
                                     feed_dict=feed_dict)
        return output
        
    def dump(self, fpath):
        model = dict()
        model['parameters'] = self.parameters
        model['n_input'] = self.n_input
        model['n_output'] = self.n_output
        model['n_step'] = self.n_step
        model['n_hidden'] = self.n_hidden
        cPickle.dump(model, open(fpath, 'wb'))
    
    def load(self, fpath):
        model = cPickle.load(open(fpath, 'rb'))
        self.parameters = model['parameters']
        self.n_input = model['n_input']
        self.n_output = model['n_output']
        self.n_hidden = model['n_hidden']
        self.n_step = model['n_step']
        self.__build_graph__(variable_values=self.parameters)


if __name__ == '__main__':
    tf.reset_default_graph()
    plt.close('all')
    np.random.seed(73)
    
    n_input = 28
    n_output = n_input
    n_step = 28
    n_hidden = 2 * n_input
    gamma = 1e-3
    learning_rate = 1e-2
    n_epoch = 10
    batch_size = 100
    validation_steps = 500

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_x = mnist.train.images.reshape([-1, 28, 28])
    print '{} training samples'.format(train_x.shape[0])
    validation_x = mnist.validation.images.reshape([-1, 28, 28])[::5, :, :]
    print '{} validation samples'.format(validation_x.shape[0])
    test_x = mnist.test.images.reshape([-1, 28, 28])
    print '{} test samples'.format(test_x.shape[0])

    model = TemporalAutoEncoder(n_input=n_input, n_step=n_step, n_hidden=n_hidden)
    model.fit(train_x=train_x, validation_x=validation_x,
              n_epoch=n_epoch, batch_size=batch_size,
              gamma=gamma, learning_rate=learning_rate, 
              validation_steps=validation_steps)
    
    model.dump('seq2seq_model.pkl')
    
    #model = TemporalAutoEncoder()
    #model.load('seq2seq_model.pkl')
    
    # evaluate on test set
    y = model.predict(test_x)
    error = np.mean((y - test_x) * (y - test_x))
    print 'test set error is {:.4f}'.format(error)

    def gray2rgb(im):
        return np.stack((im, im, im), axis=2)

    n_test_samples = 20
    selected_idx = np.random.permutation(test_x.shape[0])[0 : n_test_samples]
    x = test_x[selected_idx, :, :]
    x_output = model.predict(x)
    images = list()
    for i in range(x.shape[0]):
        images.append(x[i, :, :].reshape((1, -1)))
        images.append(x_output[i, :, :].reshape((1, -1)))
    images = np.concatenate(images, axis=0)
    im = tile_raster_images(images, [n_input, n_input], [5, n_test_samples*2/5])
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.imshow(gray2rgb(im))
    ax.axis('off')
    fig.show()
    fig.savefig('{}.png'.format(fig.number))