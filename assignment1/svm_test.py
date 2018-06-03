# -*- coding:utf-8 -*-

"""
@function: SVM线性分类器测试类
@author:HuiYi or 会意
@file:svm_test.py
@time:2018/05/27 16:31
"""
import matplotlib.pyplot as plt
from cs231n.data_utils import load_cifar10
from cs231n.classifiers.linear_svm import svm_loss_native
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.gradient_check import grad_check_sparse
import numpy as np
import time

cifar10_directory = 'F:/cs231n/assignment1/cs231n/dataset/cifar-10-batches-py'
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def get_data():
    # 读取cifar-10中的数据
    X_train, Y_train, X_test, Y_test = load_cifar10(cifar10_directory)
    # 指定不同数据集中的数量
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_development = 500
    # 验证集
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    # 训练集
    mask = range(num_training)
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    # 从训练集中随机抽取一部分的数据点作为开发集
    mask = np.random.choice(49000, num_development, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]
    # 测试集
    mask = range(num_test)
    X_test = X_test[mask]
    Y_test = Y_test[mask]
    # 将图像数据转置成二维的
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    # 预处理减去图片的平均值
    mean_image = np.mean(X_train, axis=0)  # from (49000, 3072) to (1, 3072)
    # print(mean_image[:10])  # 查看特征的数据
    # plt.figure(figsize=(4, 4))  # 制定画图的框图大小
    # plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))  # 将平均值可视化
    # plt.show()
    X_train -= mean_image
    X_test -= mean_image
    X_val -= mean_image
    X_dev -= mean_image
    # 在X中添加一列1作为偏置维度，这样在优化是只需要考虑权值矩阵W就好了
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    return X_train, Y_train, X_test, Y_test, X_val, Y_val, X_dev, Y_dev


def svm_test_native(x, y):
    # 随机生成一个很小的SVM权重矩阵，先标准正态分布，然后乘0.0001
    W = np.random.randn(3073, 10) * 0.0001
    # 计算SVM分类器的损失和权重的梯度(无正则项)
    loss, gradient = svm_loss_native(W, x, y, 0.0)
    # 随机选取W中的几个维度，计算数值梯度和解析梯度进行对比，验证正确性。 随机选取过程在gradient_check中
    # 定义一个lambda表达式，计算损失值loss,
    f = lambda w: svm_loss_native(w, x, y, 0.0)[0]
    grad_check_sparse(f, W, gradient)
    # 计算SVM分类器的损失和权重的梯度(有正则项)
    print('turn on regularization')
    loss, gradient = svm_loss_native(W, x, y, 5e1)
    # 随机选取W中的几个维度，计算数值梯度和解析梯度进行对比，验证正确性。 随机选取过程在gradient_check中
    # 定义一个lambda表达式，计算损失值loss,
    f = lambda w: svm_loss_native(w, x, y, 5e1)[0]
    grad_check_sparse(f, W, gradient)


def svm_test_vectorized(x, y):
    # 随机生成一个很小的SVM权重矩阵，先标准正态分布，然后乘0.0001
    W = np.random.randn(3073, 10) * 0.0001
    # 计算SVM分类器的损失和权重的梯度(无正则项)
    loss, gradient = svm_loss_vectorized(W, x, y, 0.0)
    # 随机选取W中的几个维度，计算数值梯度和解析梯度进行对比，验证正确性。 随机选取过程在gradient_check中
    # 定义一个lambda表达式，计算损失值loss,
    f = lambda w: svm_loss_vectorized(w, x, y, 0.0)[0]
    grad_check_sparse(f, W, gradient)
    print('turn on regularization')
    # 计算SVM分类器的损失和权重的梯度(有正则项)
    loss, gradient = svm_loss_vectorized(W, x, y, 5e1)
    # 随机选取W中的几个维度，计算数值梯度和解析梯度进行对比，验证正确性。 随机选取过程在gradient_check中
    # 定义一个lambda表达式，计算损失值loss,
    f = lambda w: svm_loss_vectorized(w, x, y, 5e1)[0]
    grad_check_sparse(f, W, gradient)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev = get_data()
    # 使用有循环的方式计算损失和梯度
    # svm_test_native(x_dev, y_dev)
    # 使用矢量化的方式计算损失和梯度
    # svm_test_vectorized(x_dev, y_dev)
    W = np.random.randn(3073, 10) * 0.0001
    # 统计两种方式的用时和计算记过差异
    tic = time.clock()
    loss_native, gradient_native = svm_loss_native(W, x_dev, y_dev, 5e1)
    toc = time.clock()
    print("Native loss and gradient: compute in %f s" % (toc - tic))
    tic = time.clock()
    loss_vectorized, gradient_vectorized = svm_loss_vectorized(W, x_dev, y_dev, 5e1)
    toc = time.clock()
    print("Vectorized loss and gradient: compute in %f s" % (toc - tic))
    print("The loss difference: %f" % (loss_native - loss_vectorized))
    # 使用Frobenius norm(弗罗贝尼乌斯范数)来比较梯度的差异  每一项绝对值平方后求和，然后开方
    grad_difference = np.linalg.norm(gradient_native - gradient_vectorized, ord="fro")
    print("The gradient difference: %f" % (loss_native - loss_vectorized))
