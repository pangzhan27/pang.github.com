
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave as ims
from copy import deepcopy
import pickle
from matplotlib import pyplot as plt


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import pickle
import cv2
import random
from Dataset_1D import DataSet_1D
from pandas import read_csv

batch_size = 113
l2_norm_loss = []


def read_data():
    norm_file = './data/Arrhythmia/norm.csv'
    anorm_file = './data/Arrhythmia/anorm.csv'
    norm = read_csv(norm_file, header=0, index_col=0)
    anorm = read_csv(anorm_file, header=0, index_col=0)
    norm_data = norm.values;
    anorm_data = anorm.values

    total_num = len(norm_data) + len(anorm_data)
    perm = np.arange(len(norm_data))
    np.random.seed(12345678)
    np.random.shuffle(perm)
    norm_data = norm_data[perm]

    threshold = int(total_num/2)
    train_x = norm_data[:int(threshold)]
    train_y = np.zeros([int(threshold),1])
    test1_x = norm_data[threshold:]
    test1_y = np.zeros([len(norm_data) - threshold, 1])
    test_x = np.row_stack((test1_x, anorm_data))
    test_y = np.row_stack((test1_y, np.ones([len(anorm_data), 1])))

    max = train_x.max(axis=0)
    min = train_x.min(axis=0)
    xx = len(train_x)
    _max = np.tile(max, (xx, 1))
    _min = np.tile(min, (xx, 1))
    diff = _max - _min
    train_xs = (train_x - _min) / (diff + 1e-6)

    xx = len(test_x)
    _max = np.tile(max, (xx, 1))
    _min = np.tile(min, (xx, 1))
    diff = _max - _min
    test_xs = (test_x - _min) / (diff + 1e-6)

    train_xs = np.clip(train_xs,0, 1.0)
    test_xs = np.clip(test_xs,-2.0, 2.0)

    all_images = np.array(train_x.tolist() + test_x.tolist())
    all_labels = np.array(train_y.tolist() + test_y.tolist())
    data = {'train': [train_x, train_y],
            'all': [all_images, all_labels],
            'test': [test_x, test_y]}
    output = open('arr.pkl', 'wb')
    pickle.dump(data, output, protocol=2)
    output.close()
    return train_xs, train_y,test_xs, test_y

def dense(x, inputFeatures, outputFeatures, scope=None, l2=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32,
                                 tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if l2 == True:
            l2_norm_loss.append(tf.nn.l2_loss(matrix))
        return tf.matmul(x, matrix) + bias

class CAE():
    def __init__(self):

        self.batchsize = batch_size
        self.images = tf.placeholder(tf.float32, [self.batchsize, 274])
        self.keep_prob = tf.placeholder(tf.float32)
        self.en = self.encoder(self.images)
        self.out = self.decoder(self.en)
        self.e = tf.reduce_sum(tf.square(self.images - self.out), axis=1, keep_dims=True) / 1

        bgn = tf.constant(1.0,tf.float32, [self.batchsize,1])
        self.lstm_in = tf.concat(1,(bgn,self.en,self.e))
        self.lstm_out = self.lstm(self.lstm_in[:,:-1])

        l2 = 0
        for i in range(len(l2_norm_loss)):
            l2 += l2_norm_loss[i]


        self.l = 0.1*tf.reduce_sum(tf.square(self.lstm_in[:,1:] - self.lstm_out), axis=1,keep_dims = True )

        self.ae = tf.reduce_mean(self.e )
        self.lstm = tf.reduce_mean( self.l)

        self.cost = self.ae +self.lstm + 0.001 * l2
        self.opt = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

    # encoder
    def encoder(self, input_images):
        with tf.variable_scope("encoder"):
            en_x1 = tf.nn.tanh(dense(input_images, 274, 100, "en_fc1"))
            en_x2 = dense(en_x1, 100, 10, "en_fc2", l2=True)
        return en_x2

    # decoder
    def decoder(self, z):
        with tf.variable_scope("decoder"):
            z1 = tf.nn.tanh(dense(z, 10, 100, scope='de_fc1'))
            z2 = dense(z1, 100, 274, scope='de_fc2')
        return z2

    def lstm(self, feature):
        feature = tf.reshape(feature, [-1, 11, 1])
        w_out = tf.get_variable("out_weights", [16, 1], tf.float32, tf.random_normal_initializer(stddev=0.5))
        b_out = tf.Variable(tf.random_normal([1]))
        input_list = tf.unstack(feature, axis=1)

        def get_a_cell(lstm_size, keep_prob):
           lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
           drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
           return drop

        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(16, self.keep_prob) for _ in range(2)])
        outputs, states = tf.nn.rnn(stacked_lstm, input_list, dtype=tf.float32)
        for i in range(len(outputs)):
           if i == 0:
               results = tf.nn.softplus(tf.matmul(outputs[i], w_out) + b_out)
           else:
               results = tf.nn.softplus(tf.concat(1,(results, (tf.matmul(outputs[i], w_out) + b_out))))
        return results

def train_cae():
    train_images,train_labels, test_images,test_labels = read_data()
    train_org = deepcopy(train_images)
    model = CAE()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs = [model.opt, model.cost, model.ae, model.lstm]
        for epoch in range(0, 1001):
            train_loss = []
            for i in range(0, len(train_images), batch_size):
                batch_images = train_images[i: i + batch_size]
                _, loss, ae, lstm = sess.run(outputs,feed_dict={model.images: batch_images,model.keep_prob: 0.5})
                train_loss.append([loss, ae, lstm])
            losses = np.mean(np.array(train_loss), axis=0)
            print('epoch{%d}, train loss: %.2f , ae loss: %.2f , lstm :%.2f'% (epoch, losses[0], losses[1], losses[2]))

            perm = np.arange(len(train_images))
            np.random.shuffle(perm)
            train_images = np.array(train_images)[perm]

            if epoch % 1000 == 0 and epoch > 0:
                pred = []
                fs = []
                label = []
                encode = []
                for i in range(0, len(test_images), batch_size):
                    batch_images = test_images[i: i + batch_size]
                    batch_labels = test_labels[i: i + batch_size]
                    ae,lstm = sess.run((model.e, model.l),feed_dict={ model.images: batch_images,model.keep_prob: 1})
                    pred += ae.tolist()
                    fs += lstm.tolist()
                    label += batch_labels.tolist()
                pred = np.array(pred)
                pred.shape = -1, 1
                label = np.array(label)
                label.shape = -1, 1
                fs = np.array(fs)
                fs.shape = -1, 1
                db = np.column_stack((pred, fs, label))
                np.save('./model_cae/' + 'test' + str(epoch) + '.npy', db)

        pred = []
        fs = []
        label = []

        for i in range(0, len(train_org), batch_size):
            batch_images = train_org[i: i + batch_size]
            batch_labels = train_labels[i: i + batch_size]
            ae, lstm = sess.run((model.e, model.l), feed_dict={model.images: batch_images, model.keep_prob: 1})
            pred += ae.tolist()
            fs += lstm.tolist()
            label += batch_labels.tolist()
        pred = np.array(pred)
        pred.shape = -1, 1
        label = np.array(label)
        label.shape = -1, 1
        fs = np.array(fs)
        fs.shape = -1, 1
        db = np.column_stack((pred, fs, label))
        np.save('./model_cae/' + 'train' + str(0000) + '.npy', db)

        pred = []
        fs = []
        label = []
        for i in range(0, len(test_images), batch_size):
            batch_images = test_images[i: i + batch_size]
            batch_labels = test_labels[i: i + batch_size]
            ae, lstm = sess.run((model.e, model.l), feed_dict={model.images: batch_images, model.keep_prob: 1})
            pred += ae.tolist()
            fs += lstm.tolist()
            label += batch_labels.tolist()
        pred = np.array(pred)
        pred.shape = -1, 1
        label = np.array(label)
        label.shape = -1, 1
        fs = np.array(fs)
        fs.shape = -1, 1
        db = np.column_stack((pred, fs, label))
        np.save('./model_cae/' + 'test' + str(0000) + '.npy', db)

Fla = 1
if __name__ == "__main__":
    if Fla == 1:
        train_cae()
    else:
        pass
