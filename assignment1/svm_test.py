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
from cs231n.classifiers.linear_classifier import LinearSVM
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
    # # 使用有循环的方式计算损失和梯度
    # svm_test_native(x_dev, y_dev)
    # # 使用矢量化的方式计算损失和梯度
    # svm_test_vectorized(x_dev, y_dev)
    # W = np.random.randn(3073, 10) * 0.0001
    # # 统计两种方式的用时和计算记过差异
    # tic = time.clock()
    # loss_native, gradient_native = svm_loss_native(W, x_dev, y_dev, 5e1)
    # toc = time.clock()
    # print("Native loss and gradient: compute in %f s" % (toc - tic))
    # tic = time.clock()
    # loss_vectorized, gradient_vectorized = svm_loss_vectorized(W, x_dev, y_dev, 5e1)
    # toc = time.clock()
    # print("Vectorized loss and gradient: compute in %f s" % (toc - tic))
    # print("The loss difference: %f" % (loss_native - loss_vectorized))
    # # 使用Frobenius norm(弗罗贝尼乌斯范数)来比较梯度的差异  每一项绝对值平方后求和，然后开方
    # grad_difference = np.linalg.norm(gradient_native - gradient_vectorized, ord="fro")
    # print("The gradient difference: %f" % (loss_native - loss_vectorized))

    # ------------------------------------------------我是分割线-------------------------------------------------- #
    # 使用随机梯度下降方法进行反向传播
    # svm = LinearSVM()
    # tic = time.clock()
    # lost_history = svm.train(x_train, y_train, learning_rate=1e-7, reg=2.5e4, num_iters=2000, batch_size=256, verbose=True)
    # toc = time.clock()
    # print('That took %fs' % (toc - tic))
    # # A useful debugging strategy is to plot the loss as a function of iteration number:
    # plt.plot(lost_history)
    # plt.xlabel('Iteration number')
    # plt.ylabel('Loss value')
    # plt.show()
    # # Write the LinearSVM.predict function and evaluate the performance on both the training and validation set
    # y_train_pred = svm.predict(x_train)
    # print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
    # y_val_pred = svm.predict(x_val)
    # print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))

    # ------------------------------------------------我是分割线-------------------------------------------------- #
    # 使用验证集去调整超参数（正则化强度和学习率），尝试不同的学习率和正则化强度
    # 可以先用较大的步长搜索，然后进行微调
    #  if you are careful you should be able to get a classification accuracy of about 0.4 on the validation set.
    learning_rates = [2.0e-7, 1.75e-7, 1.5e-7, 1.25e-7, 1e-7]
    regularization_strengths = [3.75e4, 4e4, 4.25e4]
    # 结果集为一个字典，其键的形式为(learning_rate, regularization_strength)的tuple，
    # 值的形式为(training_accuracy, validation_accuracy)的tuple
    results = {}
    best_val = -1  # 出现的正确率最大值
    best_svm = None  # 达到正确率最大值的SVM对象
    ################################################################################
    # Write code that chooses the best hyper-parameters by tuning on the validation #
    # set. For each combination of hyper-parameters, train a linear SVM on the      #
    # training set, compute its accuracy on the training and validation sets, and  #
    # store these numbers in the results dictionary. In addition, store the best   #
    # validation accuracy in best_val and the LinearSVM object that achieves this  #
    # accuracy in best_svm.                                                        #
    #                                                                              #
    # Hint: You should use a small value for num_iters as you develop your         #
    # validation code so that the SVMs don't take much time to train; once you are #
    # confident that your validation code works, you should rerun the validation   #
    # code with a larger value for num_iters.                                      #
    ################################################################################
    for rate in learning_rates:
        for regularization in regularization_strengths:
            svm = LinearSVM()
            lost_history = svm.train(x_train, y_train, rate, regularization, 2000)
            y_train_pred = svm.predict(x_train)
            accuracy_train = np.mean(y_train == y_train_pred)
            y_val_pred = svm.predict(x_val)
            accuracy_val = np.mean(y_val == y_val_pred)
            results[(rate, regularization)] = (accuracy_train, accuracy_val)
            if best_val < accuracy_val:
                best_val = accuracy_val
                best_svm = svm
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################
    # 打印结果
    for lr, reg in sorted(results):
        training_accuracy, validation_accuracy = results[(lr, reg)]
        print('lr %e reg %e training_accuracy: %f validation_accuracy %f ' %
              (lr, reg, training_accuracy, validation_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # 可视化交叉验证集结果
    import math
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]
    # plot training accuracy
    market_size = 100
    colors = [results[x][0] for x in results]
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, market_size, colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')
    # plot validation accuracy
    colors = [results[x][1] for x in results]
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, market_size, colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.tight_layout()  # 调整子图间距
    plt.show()

    # Evaluate the best svm on test set
    y_test_pred = best_svm.predict(x_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

    # Visualize the learned weights for each class.
    # Depending on your choice of learning rate and regularization strength, these may
    # or may not be nice to look at.
    w = best_svm.W[:-1, :]  # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(2)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

