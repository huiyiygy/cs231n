# -*- coding:utf-8 -*-

"""
@author:HuiYi or 会意
@file:knn_test.py
@time:2018/03/28 16:22
"""

# 测试快速排序
# import random
# import time
#
# old_data = list(random.randrange(200) for i in range(200))
# print(old_data)
#
#
# def quick_sort(array):
#     if len(array) <= 1:
#         return array
#     mid_num = array[len(array) // 2]
#     left = [x for x in array if x < mid_num]
#     middle = [x for x in array if x == mid_num]
#     right = [y for y in array if y > mid_num]
#     return quick_sort(left) + middle + quick_sort(right)
#
#
# start = time.clock()
# new_data = quick_sort(old_data)
# end = time.clock()
# print(new_data)
# print('快速排序用时：%f s' % (end-start))


# 测试numpy和matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(0, 10 * np.pi, 0.01)
# y = np.sin(x)
#
# plt.plot(x, y)
# plt.show()
# print('正弦函数图像')


# 测试闭包
# def say_hello(prefix):
#     prefix = prefix + '45679w'
#
#     def hello(name):
#         print(prefix, name)
#     return hello
#
#
# a = say_hello('good morning,')


# 测试类中的方法重载
# 类的专有方法：
# __init__ : 构造函数，在生成对象时调用
# __del__ : 析构函数，释放对象时使用
# __repr__ : 打印，转换
# __setitem__ : 按照索引赋值
# __getitem__: 按照索引获取值
# __len__: 获得长度
# __cmp__: 比较运算
# __call__: 函数调用
# __add__: 加运算
# __sub__: 减运算
# __mul__: 乘运算
# __div__: 除运算
# __mod__: 求余运算
# __pow__: 乘方
# class MyList:
#     __my_list = list()
#
#     def __init__(self, *args):
#         for arg in args:
#             self.__my_list.append(arg)
#
#     '''重载自定义加法'''
#     def __add__(self, x):
#         for i in range(0, len(self.__my_list)):
#             self.__my_list[i] += x
#
#     def show(self):
#         print(self.__my_list)
#
#
# testList = MyList(1, 2, 3, 4, 5, 6, 7)
# testList + 10
# testList.show()

import numpy as np
from data_utils import get_cifar_data
from classifiers import KNearestNeighbor
import time
import matplotlib.pyplot as plt


def time_functions(f, *args):
    """
    记录函数执行时间的装饰器
    """
    tic = time.time()
    y = f(*args)
    toc = time.time()
    return [y, toc - tic]


def verification_label(predicted, original):
    """验证标签"""
    # classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 类别列表
    correct_number = int(np.sum(predicted == original))
    num_test = original.shape[0]
    # for i in range(num_test):
    #     predicted_label_index = int(predicted[i])
    #     original_label_index = int(original[i])
    #     print('predicted label:%s,\t original label:%s\n' %
    #           (classes[predicted_label_index], classes[original_label_index]))
    print('correct number: %d, error number: %d, accuracy: %f ' %
          (correct_number, num_test - correct_number, correct_number * 1.0 / num_test))


def cross_validation(train_data, train_label):
    """交叉验证的方式选择最优的超参数k"""
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    # 任务：
    # 将训练数据切分，训练样本和对应的样本标签包含在数组
    # x_train_folds 和 y_train_folds 之中，数组的长度为num_folds
    # 其中y_train_folds[i] 是一个矢量，表示矢量x_train_folds[i]中所有样本的标签
    # 提示：可以尝试使用numpy的 array_spilt 方法
    x_train_folds = np.array_split(train_data, num_folds)
    y_train_folds = np.array_split(train_label, num_folds)
    # 我们将不同k值下的准确率保存在一个字典中。交叉验证之后，k_to_accuracies[k]保存了一个
    # 长度为num_folds的list，值为k值下的准确率
    k_to_accuracies = {}
    # 任务：
    # 通过k折的交叉验证找到最佳k值。对于每一个k值，执行KNN算法num_folds次，每一次执行中，选择一折为验证集
    # 其它折为训练集。将不同k值在不同折上的验证结果保存在k_to_accuracies字典中
    classifiers = KNearestNeighbor()
    for k in k_choices:
        accuracies = np.zeros(num_folds)
        for fold in range(num_folds):
            temp_x = x_train_folds.copy()
            temp_y = y_train_folds.copy()
            # 组成验证集
            x_validate_fold = temp_x.pop(fold)
            y_validate_fold = temp_y.pop(fold)
            # 组成训练集
            x_temp_train_fold = np.array([x for x_fold in temp_x for x in x_fold])
            y_temp_train_fold = np.array([y for y_fold in temp_y for y in y_fold])
            classifiers.train(x_temp_train_fold, y_temp_train_fold)
            # 进行验证
            y_test_predicted = classifiers.predict(x_validate_fold, k, 0)
            num_correct = np.sum(y_test_predicted == y_validate_fold)
            accuracy = float(num_correct) / y_validate_fold.shape[0]
            accuracies[fold] = accuracy
        k_to_accuracies[k] = accuracies
    # 输出准确率
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))
    # 画图显示所有的精确度散点
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k]*len(accuracies), accuracies)
    # plot the trend line with error bars that correspond to standard
    # 画出在不同k值下，误差均值和标准差
    accuracies_mean = np.array([np.mean(k_to_accuracies[k]) for k in sorted(k_to_accuracies)])
    accuracies_std = np.array([np.std(k_to_accuracies[k]) for k in sorted(k_to_accuracies)])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


if __name__ == '__main__':
    # 读取cifar-10中的数据
    data = get_cifar_data(num_training=5000, num_validation=1000, num_test=500, subtract_mean=False)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    # 将图像数据转置成二维的
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    # classifiers = KNearestNeighbor()  # 创建分类器对象
    # classifiers.train(x_train, y_train)  # 存入训练数据
    # [y_predicted, time_used] = time_functions(classifiers.predict, x_test, 3, 0)  # 预测标签
    # print('no loops version time used: %f seconds' % time_used)
    # verification_label(y_predicted, y_test)  # 验证标签
    # 交叉验证的方式选择最优的超参数k
    cross_validation(x_train, y_train)


