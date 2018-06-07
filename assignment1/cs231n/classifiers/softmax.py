# -*- coding:utf-8 -*-

"""
@function:
@author:HuiYi or 会意
@file:softmax.py
@time:2018/06/07 14:18
"""
import numpy as np


def softmax_loss_native(W, x, y, reg):
    """
    Softmax loss function, native implementation(with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches of N examples

    Inputs:
    -------
    - W: A numpy array of shape (D, C) containing weights.
    - x: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels;
        y[i] = c means that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns:
    --------
    - loss: as single float
    - grad: gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # Compute the softmax loss and its gradient using explicit loops.           #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = x.shape[0]
    for i in range(num_train):
        # 计算分值向量
        f_i = x[i].dot(W)
        # 为避免数值不稳定，每一个分值向量减去向量中的最大值
        f_i -= np.max(f_i)
        # 计算分值向量对应每个类别的概率
        sum_j = np.sum(np.exp(f_i))
        p = lambda k: np.exp(f_i[k]) / sum_j
        # 计算损失值
        loss += -np.log(p(y[i]))  # 每个图像的损失值加在一起，最后求均值
        # 计算梯度
        for k in range(num_classes):
            dW[:, k] += (p(k) - (k == y[i])) * x[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(np.square(W))
    dW /= num_train
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, x, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # Compute the softmax loss and its gradient using no explicit loops.        #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = x.shape[0]
    # 计算分值
    f = x.dot(W)
    # 为避免数值不稳定，每一个分值向量减去向量中的最大值
    f -= np.max(f, axis=1, keepdims=True)
    # 计算分值向量对应每个类别的概率
    f_exp = np.exp(f)
    sum_f = np.sum(f_exp, axis=1, keepdims=True)
    p = f_exp / sum_f
    # 计算损失
    loss += np.sum(-np.log(p[np.arange(num_train), y]))
    # 计算梯度
    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1
    dW = x.T.dot(p - ind)

    loss /= num_train
    loss += 0.5 * reg * np.sum(np.square(W))
    dW /= num_train
    dW += reg * W
    return loss, dW
