# -*- coding:utf-8 -*-

"""
@function:
@author:HuiYi or 会意
@file:k_nearest_neighbor.py
@time:2018/04/30 15:59
"""
import numpy as np


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, images, labels):
        """
        :param images: is N x D where each row is an example, N 图片个数, D 图片数据维数
        :param labels: is 1-dimension of size N
        """
        self.xtr = images
        self.ytr = labels

    def predict(self, test_images):
        """
        :param test_images: is N x D where each row is an example we wish to predict label for
        """
        num_test = test_images.shape[0]
        # lets make sure that output type matches the input type
        y_predict = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the 'i'th test image
            # using the L1 distance(sum of absolute value differences)
            # distances = np.sum(np.abs(self.xtr - test_images[i, :]), axis=1)
            # using the L2 distance
            distances = np.sqrt(np.sum(np.square(self.xtr - test_images[i, :])))
            min_index = np.argmin(distances)  # get the index with smallest distance
            y_predict[i] = self.ytr[min_index]  # predict the label of the nearest example
        return y_predict


class KNearestNeighbor:
    """a KNN classifier with L2 distance"""

    def train(self, x, y):
        """
        Train the classifier. For k-nearest neighbor this is just memorizing the training data

        Inputs
        ------
        - x: A numpy array of shape(num_train, D) containing the training data
             consisting of num_train samples each of dimension D.
        - y: A numpy array of shape(N,) containing the training labels,
             where y[i] is the label for x[i]
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs
        ------
        - x: A numpy array of shape(num_test, D) containing test data consistining of num_test
            samples each of dimension D.
        - k: The number of nearest neighbor that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances between training
            points and testing points.

        Returns
        -------
        - y: A numpy array of shape(num_test,) containing predicted labels for the test data,
            where y[i] is the predicted label for the test point x[i].
        """
        if num_loops == 0:
            distances = self.__compute_distances_no_loops(x)
        elif num_loops == 1:
            distances = self.__compute_distances_one_loops(x)
        elif num_loops == 2:
            distances = self.__compute_distances_two_loops(x)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.__predict_labels(distances, k)

    def __compute_distances_no_loops(self, x):
        """
        Compute the distance between each test point in x and each training point in
        self.x_train using no explicit loops.

        Inputs
        ------
        - x: A numpy array of shape(num_test, D) containing test data.

        Returns
        -------
        - distances: A numpy array of shape(num_test, num_train) where distances[i, j] is
            the L2(Euclidean) distance between the ith test point and the jth training point.
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        distances = np.zeros((num_test, num_train))
        # 计算所有测试点和所有训练点之间的距离，不使用任何显式循环，并将结果存储在distances中。
        # 你应该只使用基本的数组操作来实现这个函数;
        # 特别是你不应该使用scipy的函数。
        # 提示：尝试使用矩阵乘法和两个广播总和来表达l2距离。
        # sqrt(square(A-B)) = sqrt(square(A) - 2 * A * B + square(B))
        distances += -2 * np.dot(x, self.x_train.T)  # shape(num_test, num_train)
        distances += np.sum(np.square(self.x_train), axis=1).reshape(1, num_train)
        distances += np.sum(np.square(x), axis=1).reshape(num_test, 1)
        distances = np.sqrt(distances)
        return distances

    def __compute_distances_one_loops(self, x):
        """
        Compute the distance between each test point in x and each training point in
        self.x_train using a single loop over the test data.

        Inputs
        ------
        - x: A numpy array of shape(num_test, D) containing test data.

        Returns
        -------
        - distances: A numpy array of shape(num_test, num_train) where distances[i, j] is
            the L2(Euclidean) distance between the ith test point and the jth training point.
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        distances = np.zeros((num_test, num_train))
        # 计算第i个测试点与所有训练点之间的距离，并将结果存储在dists中[i，：]
        for i in range(num_test):
            distances[i, :] = np.sqrt(np.sum(np.square(self.x_train - x[i]), axis=1))

        return distances

    def __compute_distances_two_loops(self, x):
        """
        Compute the distance between each test point in x and each training point in
        self.x_train using a nested loop over both the training data and the test data.

        Inputs
        ------
        - x:A numpy array of shape(num_test, D) containing test data.

        Returns
        -------
        - distances:A numpy array of shape(num_test, num_train) where distances[i, j] is
            the L2(Euclidean) distance between the ith test point and the jth training point.
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                distances[i][j] = np.sqrt(np.sum(np.square(self.x_train[j] - x[i])))

        return distances

    def __predict_labels(self, distances, k=1):
        """
        Given a matrix of distances between test points and training points, predict a label
        for each point.

        Inputs
        ------
        - distances: A numpy array of shape(num_test, num_train) where distances[i, j] gives
            the distance between the ith test point and the jth training point.
        - k:

        Returns
        -------
        - y: A numpy  array of shape(num_test,) containing predicted labels for test data, where
            y[i] is the predicted label for the test point x[i]
        """
        num_test = distances.shape[0]
        y_predicted = np.zeros(num_test)
        for i in range(num_test):
            # 长度为k的列表存储第i个测试点的k个最近邻的标签。
            closest_y = []
            # 使用距离矩阵查找第i个测试点的k个最近邻居，并使用self.y_train查找这些邻居的标签。
            # 将这些标签存储在closest_y中。
            # 提示：查找函数numpy.argsort。 返回的是数组值从小到大的索引值
            index = np.argsort(distances[i])
            closest_y = self.y_train[index[:k]]
            # 现在您已经找到了k个最近邻居的标签，您需要在标签的closest_y列表中找到最常见的标签。
            # 将此标签存储在y_pred[i]中。通过选择较小的标签打破关系。
            # np.argmax()返回沿轴axis最大值的索引
            # np.bincount(A) 假设A中最大值为a(非负数)，则返回长度为a+1的列表，索引范围为0~a，列表中的值分别表示0~a在A中出现的次数
            # 详见https://blog.csdn.net/xlinsist/article/details/51346523
            y_predicted[i] = np.argmax(np.bincount(closest_y))

        return y_predicted
