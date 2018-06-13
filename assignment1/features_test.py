# -*- coding:utf-8 -*-

"""
@function:
@author:HuiYi or 会意
@file:features_test.py
@time:2018/06/13 14:39
"""
import random
import numpy as np
import matplotlib.pyplot as plt

from cs231n.data_utils import load_cifar10
from cs231n.features import *
from cs231n.classifiers.linear_classifier import LinearSVM
from cs231n.classifiers.neural_net import TwoLayerNet


def get_data():
    cifar10_directory = 'F:/cs231n/assignment1/cs231n/dataset/cifar-10-batches-py'
    # 读取cifar-10中的数据
    X_train, Y_train, X_test, Y_test = load_cifar10(cifar10_directory)
    # 指定不同数据集中的数量
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    # 验证集
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    # 训练集
    mask = range(num_training)
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    # 测试集
    mask = range(num_test)
    X_test = X_test[mask]
    Y_test = Y_test[mask]
    return X_train, Y_train, X_test, Y_test, X_val, Y_val


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_val, y_val = get_data()

    num_color_bins = 10  # Number of bins in the color histogram
    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
    X_train_feats = extract_features(X_train, feature_fns, verbose=True)
    X_val_feats = extract_features(X_val, feature_fns)
    X_test_feats = extract_features(X_test, feature_fns)

    # 预处理：减去每一列特征的平均值
    mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
    X_train_feats -= mean_feat
    X_val_feats -= mean_feat
    X_test_feats -= mean_feat

    # 预处理: 每一列除以标准差，确保每个特征都在一个数值范围内
    std_feat = np.std(X_train_feats, axis=0, keepdims=True)
    X_train_feats /= std_feat
    X_val_feats /= std_feat
    X_test_feats /= std_feat

    # 多加一个bias列 用于SVM训练中的偏置
    X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
    X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
    X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

    # ------------------------------------------------我是分割线-------------------------------------------------- #
    # SVM训练
    # 用前面作业中的多分类SVM来对上面抽取到的特征进行训练，这次应该会比之前直接在像素上训练的结果更好
    # learning_rates = [1e-9, 1e-8, 1e-7]
    # regularization_strengths = [5e4, 5e5, 5e6]
    #
    # results = {}
    # best_val = -1
    # best_svm = None
    # ################################################################################
    # # 用验证集来调整 learning rate 和 regularization 的强度. #
    # # This should be identical to the validation that you did for the SVM; save    #
    # # the best trained classifer in best_svm. You might also want to play          #
    # # with different numbers of bins in the color histogram. If you are careful    #
    # # you should be able to get accuracy of near 0.44 on the validation set.       #
    # ################################################################################
    # for rs in regularization_strengths:
    #     for lr in learning_rates:
    #         svm = LinearSVM()
    #         loss_hist = svm.train(X_train_feats, y_train, lr, rs, 2000)
    #         y_train_pred = svm.predict(X_train_feats)
    #         accuracy_train = np.mean(y_train == y_train_pred)
    #         y_val_pred = svm.predict(X_val_feats)
    #         accuracy_val = np.mean(y_val == y_val_pred)
    #         results[(lr, rs)] = (accuracy_train, accuracy_val)
    #         if best_val < accuracy_val:
    #             best_val = accuracy_val
    #             best_svm = svm
    # ################################################################################
    # #                              END OF YOUR CODE                                #
    # ################################################################################
    # # Print out results.
    # for lr, reg in sorted(results):
    #     train_accuracy, val_accuracy = results[(lr, reg)]
    #     print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
    #         lr, reg, train_accuracy, val_accuracy))
    #
    # print('best validation accuracy achieved during cross-validation: %f' % best_val)
    #
    # # Evaluate your trained SVM on the test set
    # y_test_pred = best_svm.predict(X_test_feats)
    # test_accuracy = np.mean(y_test == y_test_pred)
    # print(test_accuracy)
    #
    # # 想知道算法是如何运作的很重要的方法是把它的分类错误可视化。在这里的可视化中，我们展示了我们系统
    # # 错误分类的图片。比如第一列展示的是实际label为飞机，但被我们系统错误标注成其他label的图片
    # examples_per_class = 8
    # classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # for cls, cls_name in enumerate(classes):
    #     idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    #     idxs = np.random.choice(idxs, examples_per_class, replace=False)
    #     for i, idx in enumerate(idxs):
    #         plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
    #         plt.imshow(X_test[idx].astype('uint8'))
    #         plt.axis('off')
    #         if i == 0:
    #             plt.title(cls_name)
    # plt.show()

    # ------------------------------------------------我是分割线-------------------------------------------------- #
    # Neural Network on image features
    # training a neural network on image features. This approach should outperform all
    # previous approaches: you should easily be able to achieve over 55% classification
    # accuracy on the test set; our best model achieves about 60% classification accuracy.
    # 预处理：减去bias列
    X_train_feats = X_train_feats[:, :-1]
    X_val_feats = X_val_feats[:, :-1]
    X_test_feats = X_test_feats[:, :-1]

    input_dim = X_train_feats.shape[1]
    hidden_dim = 500
    num_classes = 10

    results = {}
    best_net = None
    best_val_acc = -1
    ################################################################################
    # Train a two-layer neural network on image features. You may want to    #
    # cross-validate various parameters as in previous sections. Store your best   #
    # model in the best_net variable.                                              #
    ################################################################################
    learning_rate = [1e-1, 5e-1, 1, 1.5, 2]
    regularization_strength = [1e-3, 5e-3, 7.5e-3, 1e-2]
    print('running...')
    for lr in learning_rate:
        for reg in regularization_strength:
            print('.')
            net = TwoLayerNet(input_dim, hidden_dim, num_classes)
            # Train the network
            stats = net.train(X_train_feats, y_train, X_val_feats, y_val, num_iters=1500, batch_size=200, learning_rate=lr,
                              learning_rate_decay=0.95, reg=reg, verbose=False)
            val_acc = np.mean(net.predict(X_val_feats) == y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net
            results[(lr, reg)] = val_acc

    print('finished')
    # Print out results.
    for lr, reg in sorted(results):
        val_acc = results[(lr, reg)]
        print('lr %e reg %e val accuracy: %f' % (lr, reg, val_acc))

    print('best validation accuracy achieved during cross-validation: %f' % best_val_acc)
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################

    test_acc = (best_net.predict(X_test_feats) == y_test).mean()
    print(test_acc)
