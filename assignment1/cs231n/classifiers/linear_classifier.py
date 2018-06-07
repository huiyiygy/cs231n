# -*- coding:utf-8 -*-

"""
@function:
@author:HuiYi or 会意
@file:linear_classifier.py
@time:2018/06/06 15:33
"""
import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, x, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=256,
              verbose=False):
        """
        Train this linear classifier using stochastic gradient descent随机梯度下降SGD

        Inputs:
        -------
        - x: A numpy array of shape (N, D) containing training data; there are N training
            samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means that x[i]
            has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimization
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) if true, print progress during optimization.

        Returns:
        --------
        - loss_history: A list containing the value of the loss function at each training iteration
        """
        num_train, dim = x.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient to optimize W
        loss_history = []
        for it in range(num_iters):
            x_batch = None
            y_batch = None
            #########################################################################
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            batch_inx = np.random.choice(num_train, batch_size)
            x_batch = x[batch_inx, :]
            y_batch = y[batch_inx]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(x_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.W = self.W - learning_rate * grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for data points.

        Inputs:
        -------
        - x: A numpy array of shape (N, D) containg training data; there are N training samples
            each of dimension D.

        Returns:
        --------
        - y_pred: Predicted labels for the data in x. y_pred is a 1-dimensional array of length
            N, and each element is an integer giving the predicted class.
        """
        y_pred = np.zeros(x.shape[0])
        ###########################################################################
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores = x.dot(self.W)
        y_pred = np.argmax(scores, axis=1)  # 返回沿轴axis最大值的索引
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, x_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclass will override this.

        Inputs:
        -------
        - x_batch: A numpy array of shape (N, D) containing a minibatch of N data points;
            each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns:
        --------
        - loss: as a single float
        - grad: gradient with respect to self.W; an array of the same shape as W
        """
        loss = 0.0
        grad = np.zeros_like(self.W.shape)
        return loss, grad


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, x_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, x_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, x_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, x_batch, y_batch, reg)
