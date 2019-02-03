from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import copy
import numpy as np
import pandas as pd
import math 

from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from load_mnist import load_mnist

from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.image_utils import plot_flat_bwimage, plot_flat_colorimage, plot_flat_colorgrad

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial


def pcaVis(dtm, colors, centroids, title):
    pca = PCA(n_components=2).fit(dtm)
    reduced_data = pca.transform(dtm)
    reduced_centroids = pca.transform(centroids)
    plt.gcf().clear()
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors)
    plt.title("PCA Reduced Clustering Visualization " + title)
    plt.colorbar()
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1],
        marker='x', s=169, linewidths=3,
        color='r', zorder=10)
    plt.show()


def getDistancesbyCluster(dt, cluster_indices, centroids, min_distance=None, max_distance=None):
    distances = {}
    # loop through all
    for i in xrange(dt.shape[0]):
        # find which cluster its in
        cluster = cluster_indices[i]
        if not distances.has_key(cluster):
            distances[cluster] = {}
        distances[cluster][i] = getDistance(dt[i], centroids[cluster])
    # Calculate min/max distances if none were passed in
    if min_distance == None:
        min_distance = {}
        for cluster in distances:
            min_distance[cluster] = min(distances[cluster].values())
    if max_distance == None:
        max_distance = {}
        for cluster in distances:
            max_distance[cluster] = max(distances[cluster].values())
    for cluster in distances:
        min_dist = float(min_distance[cluster])
        max_dist = float(max_distance[cluster])
        # if min_dist == max_dist:
        #     print(cluster)
        #     print max_dist
        #     print min_dist
        #     print len(distances[cluster])
        for c, dist in distances[cluster].items():
            distances[cluster][c] = (dist - min_dist) / (max_dist - min_dist)
    return distances, min_distance, max_distance 

def getDistance(p1, p2):
    """
    Takes in two point arrays in the same dimension, returns the euclidean distance between them
    """
    if p1.size != p2.size:
        # print "Vector sizes do not match!"
        # print p1
        # print p2
        raise ValueError()
    sums = 0
    for i in range(p1.size):
        sums += (p1[i] - p2[i])**2
    return math.sqrt(sums)

def flattenDict(dic):
    """
    Takes in a dictionary of dictionaries, returns just a dictionary, ignoring the outer 
    level of keys and unioning the inner values together
    """
    newDict = {}
    for key in dic.keys():
        newDict.update(dic[key])
    return newDict

def getClosest(distances):
    #Returns indices of farthest
    closest = {}
    for cluster in distances:
        closest[cluster] = min(distances[cluster].iterkeys(), key=(lambda key: distances[cluster][key]))
    return closest;

def isPositive(row):
    isPositive = True
    for i in range(len(row)):
        if row[i] <= 0:
            isPositive = False
    return isPositive

def notIsNegative(row):
    isNegative = True
    for i in range(len(row)):
        if row[i] > 0:
            isNegative = False
    return not isNegative

def summing(row):
    return np.sum(row) > 0






# START HERE


data_sets = load_mnist('data')

pos_class = 4
neg_class = 9

x_train = data_sets.train.x
y_train = data_sets.train.labels
x_test = data_sets.test.x
y_test = data_sets.test.labels

x_train, y_train = dataset.filter_dataset(x_train, y_train, pos_class, neg_class)
x_test, y_test = dataset.filter_dataset(x_test, y_test, pos_class, neg_class)

#print(len(x_train), len(x_train[y_train == 1]), len(x_train[y_train == -1]))
# 10761 5307 5454


X_fold1 = x_train[:3587]
X_fold2 = x_train[3587:7174]
X_fold3 = x_train[7174:]

Y_fold1 = y_train[:3587]
Y_fold2 = y_train[3587:7174]
Y_fold3 = y_train[7174:]


X_train1 = np.concatenate((X_fold1, X_fold2))
X_train2 = np.concatenate((X_fold1, X_fold3))
X_train3 = np.concatenate((X_fold2, X_fold3))

Y_train1 = np.concatenate((Y_fold1, Y_fold2))
Y_train2 = np.concatenate((Y_fold1, Y_fold3))
Y_train3 = np.concatenate((Y_fold2, Y_fold3))


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
col = 0

influences_on_7 = np.zeros((7174, 3))
influences_on_1 = np.zeros((7174, 3))

for X_train, Y_train, X_test, Y_test in zip([X_train1, X_train2, X_train3], [Y_train1, Y_train2, Y_train3], [X_fold3, X_fold2, X_fold3], [Y_fold3, Y_fold2, Y_fold3]):

    print("HERE" + str(col))
    lr_train = DataSet(X_train, np.array((Y_train + 1) / 2, dtype=int))
    lr_validation = None
    lr_test = DataSet(X_test, np.array((Y_test + 1) / 2, dtype=int))
    lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

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

    class_wise_7 = X_test[Y_test == -1]
    class_wise_1 = X_test[Y_test == 1]

    numClusters = 1
    randomState = 0
    fittedKmeans_7 = KMeans(n_clusters=numClusters, random_state=randomState).fit(class_wise_7)
    clusters_7 = fittedKmeans_7.predict(class_wise_7)
    centroids_7 = fittedKmeans_7.cluster_centers_
    #pcaVis(class_wise_7, clusters_7, centroids_7, "for nines")
    distanceScoresByCluster_7, minDist_7, maxDist_7 = getDistancesbyCluster(class_wise_7, clusters_7, centroids_7)
    closest_7_idx = getClosest(distanceScoresByCluster_7)[0]
    # plt.figure(1)
    # plt.imshow(class_wise_7[closest_7_idx, :].reshape(28,-1))
    # plt.title("centroid 9")
    # plt.show()

    mask_neg = Y_test == -1
    k = -1
    n = -1 
    while k < closest_7_idx:
        n = n + 1
        if mask_neg[n]:
            k = k + 1

    test_idx_7 = n
    num_train = len(tf_model.data_sets.train.labels)

    influences_7 = tf_model.get_influence_on_test_loss(
        [test_idx_7], 
        np.arange(len(tf_model.data_sets.train.labels)),
        force_refresh=True) * num_train


    fittedKmeans_1 = KMeans(n_clusters=numClusters, random_state=randomState).fit(class_wise_1)
    clusters_1 = fittedKmeans_1.predict(class_wise_1)
    centroids_1 = fittedKmeans_1.cluster_centers_
    #pcaVis(class_wise_1, clusters_1, centroids_1, "for fours")
    distanceScoresByCluster_1, minDist_1, maxDist_1 = getDistancesbyCluster(class_wise_1, clusters_1, centroids_1)
    closest_1_idx = getClosest(distanceScoresByCluster_1)[0]
    # plt.figure(2)
    # plt.imshow(class_wise_1[closest_1_idx, :].reshape(28,-1))
    # plt.title("centroid 4")
    # plt.show()

    mask_pos = Y_test == 1
    k = -1
    n = -1 
    while k < closest_1_idx:
        n = n + 1
        if mask_pos[n]:
            k = k + 1

    test_idx_1 = n

    influences_1 = tf_model.get_influence_on_test_loss(
        [test_idx_1], 
        np.arange(len(tf_model.data_sets.train.labels)),
        force_refresh=True) * num_train

    l7 = len(influences_7)
    l1 = len(influences_1)
    influences_on_7[:l7, col] = influences_7
    influences_on_1[:l1, col] = influences_1

    col = col + 1

fold1_infl = np.concatenate((influences_on_7[:3587, :2], influences_on_1[:3587, :2]), axis=1)
print(influences_on_7.shape, influences_on_1.shape)
fold2_infl = np.concatenate((influences_on_7[3587:, 0][:,np.newaxis], influences_on_7[:3587, 2][:,np.newaxis], influences_on_1[3587:, 0][:,np.newaxis], influences_on_1[:3587, 2][:,np.newaxis]), axis=1)
#second = np.concatenate((influences_on_1[3965:, 0], influences_on_1[:3965, 2]), axis=1)
#fold2_infl = np.concatenate((first, second), axis=1)
#fold2_infl = np.concatenate((np.concatenate((influences_on_7[3965:, 0], influences_on_7[:3965, 2]), axis=1), np.concatenate((influences_on_1[3965:, 0], influences_on_1[:3965, 2]), axis=1)) axis=1)
fold3_infl = np.concatenate((influences_on_7[3587:, 1:], influences_on_1[3587:, 1:]), axis=1)

print(fold1_infl.shape, fold2_infl.shape, fold3_infl.shape)
train_infl = np.vstack((fold1_infl, fold2_infl, fold3_infl))
#mask_pos_infl = [notIsNegative(row) for row in train_infl]
mask_pos_infl = [isPositive(row) for row in train_infl]


X_train = x_train[mask_pos_infl]
Y_train = y_train[mask_pos_infl]

print(train_infl.shape[0], len(Y_train), len(y_test))

print("REMOVED STUFF")

lr_train = DataSet(X_train, np.array((Y_train + 1) / 2, dtype=int))
lr_validation = None
lr_test = DataSet(x_test, np.array((y_test + 1) / 2, dtype=int))
lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

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