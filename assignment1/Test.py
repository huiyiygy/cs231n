# -*- coding:utf-8 -*-

"""
@author:HuiYi or 会意
@file:Test.py
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
from classifiers.k_nearest_neighbor import KNearestNeighbor


# 读取cifar-10中的数据
data = get_cifar_data(num_training=5000, num_validation=1000, num_test=500, subtract_mean=False)
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
# 将图像数据转置成二维的
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
classifiers = KNearestNeighbor()  # 创建分类器对象
classifiers.train(x_train, y_train)  # 存入训练数据
# classifiers.predict(x_test, 3)

