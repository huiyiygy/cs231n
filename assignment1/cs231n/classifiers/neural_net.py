# -*- coding:utf-8 -*-

"""
@function:
@author:HuiYi or 会意
@file:neural_net.py
@time:2018/06/08 17:08
"""
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of N, a hidden
    layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the  weight
    matrices. The network uses a ReLU non-linearity after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The output of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and biases are
        initialized to zero. Weights and biases are stored in the variable self.params, which
        is a dictionary with the follwing keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        -------
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - std:
        """
        self.params = {'W1': std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural network.

        Inputs:
        -------
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        --------
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack varibles from params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        #############################################################################
        # Perform the forward pass, computing the class scores for the input.       #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        z1 = X.dot(W1) + b1
        # ReLU激活函数 np.maximum 逐位与0比较，取其大者
        a1 = np.maximum(0, z1)
        scores = a1.dot(W2) + b2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        #############################################################################
        # Finish the forward pass, and compute the loss. This should include        #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # 为避免数值不稳定，每一个分值向量减去向量中的最大值
        scores -= np.max(scores, axis=1, keepdims=True)
        # 计算分值向量对应每个类别的概率
        exp_scores = np.exp(scores)
        sum_scores = np.sum(exp_scores, axis=1, keepdims=True)
        p = exp_scores / sum_scores
        # 计算损失
        loss = np.sum(-np.log(p[np.arange(N), y]))
        loss = loss / N + 0.5 * reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # Compute the backward pass, computing the derivatives of the weights       #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # 计算分值的梯度
        dscore = p
        dscore[np.arange(N), y] -= 1
        dscore /= N

        # 计算W2和b2
        dW2 = a1.T.dot(dscore)
        db2 = np.sum(dscore, axis=0)
        grads['W2'] = dW2
        grads['b2'] = db2

        # 计算隐藏层的输出a1的梯度
        da1 = np.dot(dscore, W2.T)

        # 计算激活函数ReLU的梯度
        dReLU = da1
        dReLU[a1 <= 0] = 0

        # 计算W1和b1
        dW1 = X.T.dot(dReLU)
        db1 = np.sum(dReLU, axis=0)
        grads['W1'] = dW1
        grads['b1'] = db1

        # 权值矩阵加上正则化惩罚
        grads['W1'] += reg * W1
        grads['W2'] += reg * W2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        -------
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
            X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
                after each epoch. 学习率衰减参数
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: if true print progress during optimization.

        Returns: a dictionary contain loss_history, train_acc_history, val_acc_history
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(1, int(num_train / batch_size))  # SGD迭代完所有的图片所需要的理论最小次数

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # Create a random minibatch of training data and labels, storing        #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            batch_inx = np.random.choice(num_train, batch_size)
            X_batch = X[batch_inx, :]
            y_batch = y[batch_inx]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # Use the gradients in the grads dictionary to update the               #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # check accuracy
                train_acc = np.mean(self.predict(X_batch) == y_batch)
                val_acc = np.mean(self.predict(X_val) == y_val)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        -------
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to classify.

        Returns:
        --------
        -y_pred: A numpy array of shape (N,) giving predicted labels for each of
            the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
            to have class c, where 0 <= c < C.
        """
        ###########################################################################
        # Implement this function; it should be VERY simple!                #
        ###########################################################################
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1)
        scores = a1.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)  # 返回沿轴axis最大值的索引
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################
        return y_pred
