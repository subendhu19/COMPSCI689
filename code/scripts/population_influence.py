from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals  

import copy
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from load_mnist import load_mnist

from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.image_utils import plot_flat_bwimage, plot_flat_colorimage, plot_flat_colorgrad

from scipy import signal

from scipy.special import expit as expit
import matplotlib.pyplot as plt

data_sets = load_mnist('data')

pos_class = 1
neg_class = 7
test_idx = 5

X_train = data_sets.train.x
Y_train = data_sets.train.labels
X_test = data_sets.test.x
Y_test = data_sets.test.labels

X_train, Y_train = dataset.filter_dataset(X_train, Y_train, pos_class, neg_class)
X_test, Y_test = dataset.filter_dataset(X_test, Y_test, pos_class, neg_class)

lr_train = DataSet(X_train, np.array((Y_train + 1) / 2, dtype=int))
lr_validation = None
lr_test = DataSet(X_test, np.array((Y_test + 1) / 2, dtype=int))
lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

num_classes = 2
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels
weight_decay = 0.01
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

num_params = 784

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=lr_data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist-17_logreg')

tf_model.train()

num_train = len(tf_model.data_sets.train.labels)

influences = tf_model.get_influence_on_test_loss(
    [test_idx], 
    np.arange(len(tf_model.data_sets.train.labels)),
    force_refresh=True) * num_train

train_7s = X_train[Y_train == -1]
train_1s = X_train[Y_train == 1]

num_7s = len(train_7s)

train_7s_yaxis = train_7s.reshape((num_7s,28,28)).sum(axis=2)

peaks = [None]*num_7s
for i in range(num_7s):
    data = train_7s_yaxis[i]
    data = data - np.mean(data)


    window = signal.general_gaussian(5, p=1, sig=10)
    window_2 = signal.general_gaussian(2, p=1, sig=10)
    filtered = signal.fftconvolve(window, data)
    filtered = signal.fftconvolve(window_2, filtered)

    cut_off = np.max(filtered)/2.0
    filtered[filtered<cut_off]=0.0
    peakidx = signal.argrelextrema(filtered,np.greater)[0]

    peaks[i] = len(peakidx)

neg_threshold = np.amin(influences)/2
good_influence_mask = influences > neg_threshold
print "Negative threshold: ", neg_threshold
print "Influence of bad points: ", np.sum(influences) - np.sum(influences[good_influence_mask])

removed_example_1 = X_train[~good_influence_mask][-1].reshape(28,-1)
removed_example_2 = X_train[~good_influence_mask][-2].reshape(28,-1)

new_better_X_train = X_train[good_influence_mask]
new_better_Y_train = np.array((Y_train[good_influence_mask] + 1) / 2, dtype=int)
print "Number of bad points removed: ", len(X_train) - len(new_better_X_train)

influences_sevens = influences[Y_train == -1]
influences_hifen_sevens = influences_sevens[np.array(peaks) == 2]
new_train_7s = train_7s[np.array(peaks) != 2]

new_X_train = np.concatenate((train_1s, new_train_7s), axis=0)
new_Y_train = np.concatenate((np.ones(len(train_1s)), np.zeros(len(new_train_7s))), axis=0)

total_population_influence = np.sum(influences_hifen_sevens)

print "Influence of the population", total_population_influence
print "Test class", Y_test[test_idx]
print "Probability score before removing the population", 1 - expit(np.array(tf_model.return_params()).dot(X_test[test_idx]))
tf_model.update_train_x_y(new_X_train, new_Y_train)
tf_model.train()
print "Probability score after removing the population", 1 - expit(np.array(tf_model.return_params()).dot(X_test[test_idx]))
plt.imshow(X_test[test_idx].reshape(28,-1))
plt.show()
tf_model.update_train_x_y(new_better_X_train, new_better_Y_train)
tf_model.train()
print "Probability score after removing the population", 1 - expit(np.array(tf_model.return_params()).dot(X_test[test_idx]))
plt.figure(1)
plt.imshow(removed_example_1)
plt.figure(2)
plt.imshow(removed_example_2)
plt.show()

