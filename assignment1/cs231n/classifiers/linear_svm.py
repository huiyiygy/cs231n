# -*- coding:utf-8 -*-

"""
@function: SVM线性分类器
@author:HuiYi or 会意
@file:linear_svm.py
@time:2018/05/27 17:23
"""
import numpy as np


def svm_loss_native(W, x, y, reg):
    """
    Structured SVM loss function native implementation(with loop)

    Inputs have dimension D, there are C classes, and we operate on minibatches of N examples.

    Inputs:
    -------
    - W: A numpy array of shape (D, C) containing weights.
    - x: A numpy array of shape (N, C) containing a minibatch of data
    - y: A numpy array of shape (N,) containing training labels; y[i] = c mean that x[i] has
        label c, where 0 <=c < C.
    - reg: (float) regularization strength 正则化强度

    Returns:
    --------
    - loss: as single float
    - gradient: with respect to weights W; an array of same shape as W 相对于权重W的梯度
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = x.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = x[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] += -x[i, :].T
                dW[:, j] += x[i, :].T

    # Right now the loss is a sum of over all training examples, but we want it to be
    # an average instead so we divide by num_train 平均损失
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss 加上L2范数正则化惩罚
    loss += reg * np.sum(np.square(W))
    dW += reg * W
    #############################################################################
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    # 计算损失函数的梯度并将其存储为dW。比起先计算损失，再计算导数，在计算损失的同时计算 #
    # 导数可能会更简单。 因此您可能需要修改上面的一些代码来计算渐变。                  #
    #############################################################################
    return loss, dW


def svm_loss_vectorized(W, x, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs have dimension D, there are C classes, and we operate on minibatches of N examples.

    Inputs:
    -------
    - W: A numpy array of shape (D, C) containing weights.
    - x: A numpy array of shape (N, C) containing a minibatch of data
    - y: A numpy array of shape (N,) containing training labels; y[i] = c mean that x[i] has
        label c, where 0 <=c < C.
    - reg: (float) regularization strength 正则化强度

    Returns:
    --------
    - loss: as single float
    - gradient: with respect to weights W; an array of same shape as W 相对于权重W的梯度
    """
    # initialize the loss and gradient as zero
    loss = 0.0
    dW = np.zeros(W.shape)
    #############################################################################
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = x.dot(W)
    num_classes = W.shape[1]
    num_train = x.shape[0]
    # 找出得分正确的项
    scores_correct = scores[np.arange(num_train), y]  # shape (500,)
    scores_correct = np.reshape(scores_correct, (num_train, -1))  # change shape to (500,1)
    margins = scores - scores_correct + 1  # note delta = 1
    margins = np.maximum(0, margins)  # 逐位比较取其大者 运用广播机制
    margins[np.arange(num_train), y] = 0  # 让正确分类的那一项的得分间隔为0
    # 计算损失
    loss += np.sum(margins) / num_train
    loss += reg * np.sum(np.square(W))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    #############################################################################
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    margins[margins > 0] = 1
    rows_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -rows_sum
    dW += np.dot(x.T, margins) / num_train + reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW

