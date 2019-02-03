from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import copy
import numpy as np
import random 
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from load_mnist import load_mnist

from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.image_utils import plot_flat_bwimage, plot_flat_colorimage, plot_flat_colorgrad

import matplotlib.pyplot as plt
import matplotlib.cm as cm


data_sets = load_mnist('data')

X_train = data_sets.train.x
Y_train = data_sets.train.labels
X_test = data_sets.test.x
Y_test = data_sets.test.labels

num_classes = 10

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.01
batch_size = 1400
initial_learning_rate = 0.001 
keep_probs = None
max_lbfgs_iter = 1000
decay_epochs = [1000, 10000]

tf.reset_default_graph()

tf_model = LogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist_logreg_lbfgs')

tf_model.train()

#best_image_label = [3]

best_image_label = [7,5,0,1,9,9,1,3,4,3]

for j in range(7,8):
    mask = Y_test == j
    k = -1
    n = -1 
    while k < best_image_label[j]:
        n = n + 1
        if mask[n]:
            k = k + 1

    test_idx = n
    plt.figure(1)
    plt.title("Test 7 apparently")
    plt.imshow(tf_model.data_sets.test.x[test_idx].reshape(28,-1))
    plt.show()
    print(j, tf_model.data_sets.test.labels[test_idx])
    num_train = len(tf_model.data_sets.train.labels)

    mask7 = data_sets.train.labels == 7
    class_wise_7 = X_train[mask7]

    influences = tf_model.get_influence_on_test_loss(
        [test_idx], 
        np.arange(len(tf_model.data_sets.train.labels)),
        force_refresh=True) * num_train

    x_test = tf_model.data_sets.test.x[test_idx, :]
    y_test = tf_model.data_sets.test.labels[test_idx]
    flipped_idx = tf_model.data_sets.train.labels != y_test

    test_image = tf_model.data_sets.test.x[test_idx, :]

    # Find a training image that has the same label but is harmful
    train_idx = np.where(~flipped_idx)[0][np.argsort(influences[~flipped_idx])[4]]
    harmful_train_image = tf_model.data_sets.train.x[train_idx, :]

    #Plot Influence vs Euclidean Distance of Training Images from Test Image
    # diff = tf_model.data_sets.train.x - test_image
    # euclidean_dist = np.sqrt(np.sum(diff**2, axis=1))
    # colors = ['red', 'green', 'yellow', 'orange', 'blue', 'purple', 'grey', 'pink', 'brown', 'black']
    # plt.gcf().clear()
    # for i in range(10):
    #     euclidean_dist_i = euclidean_dist[tf_model.data_sets.train.labels == i]
    #     influences_i = influences[tf_model.data_sets.train.labels == i]
    #     plt.scatter(euclidean_dist_i, influences_i, c = colors[i], label = str(i))
    # plt.title("Influence vs Euclidean Dist, Test in Class " + str(j))
    # plt.legend(loc='upper right')
    # plt.xlabel("Euclidean Distance")
    # plt.ylabel("Influence")
    # plt.show()

    #training image that most negatively influences test image
    #neg_mask = [influences < 0]
    # neg_training = tf_model.data_sets.train.x[neg_mask and mask7]
    # neg_training_labels = tf_model.data_sets.train.labels[neg_mask and mask7]
    infl_7 = influences[mask7]
    for i in range(len(class_wise_7)):
        if infl_7[i] < 0:
            plt.figure(i)
            plt.title("index " + str(i))
            plt.imshow(class_wise_7[i, :].reshape(28, -1))
            plt.show()
        i = i + 1


    # max_idx = np.argmax(influences)
    # plt.figure(j)
    # plt.title("most pos influence")
    # plt.imshow(tf_model.data_sets.train.x[max_idx, :].reshape(28, -1))
    # plt.show()

    # min_idx = np.argmin(influences)
    # plt.figure(j)
    # plt.title("most neg influence")
    # plt.imshow(tf_model.data_sets.train.x[min_idx, :].reshape(28, -1))
    # plt.show()


# np.savez('output/components_results',
#     influences=influences,
#     # influences_without_train_error=influences_without_train_error,
#     # influences_without_hessian=influences_without_hessian,
#     # influences_without_both=influences_without_both,
#     flipped_idx=flipped_idx,
#     test_image=test_image,
#     harmful_train_image=harmful_train_image)