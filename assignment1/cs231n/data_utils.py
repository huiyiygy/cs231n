# -*- coding:utf-8 -*-

"""
@function:加载cifar10数据集模块
@author:HuiYi or 会意
@file:data_utils.py
@time:2018/04/27 13:57
"""
import matplotlib.image as plimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
import platform


def __load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def __load_cifar_batch(filename):
    """
    Load single batch from cifar-10

    Inputs
    ------
    - filename:batch的文件名

    Returns
    -------
    - x:读取的图像数据，格式(10000, 32, 32, 3)
    - y:图像数据对应的标签
    """
    with open(filename, 'rb') as f:
        datadict = __load_pickle(f)
        x = datadict['data']  # x的size为(10000, 3072(3*32*32))
        y = datadict['labels']  # y的格式为list
        # x = x.reshape(10000, 3, 32, 32)
        # 转换轴，使得从(10000, 3, 32, 32) 变为(10000, 32, 32, 3)（后面代码又把它转为了(3,32,32)，总感觉是多此一举╮(╯▽╰)╭）
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y = np.array(y)
        return [x, y]


def load_cifar10(cifar10_path):
    """
    读取cifar-10数据集，将data_batch_1到5中的所有数据作为训练集,test_batch中的数据作为测试集

    Inputs
    ------
    - cifar10_dir:数据集所在路径

    Returns
    -------
    - x_train:训练数据
    - y_train:训练数据标签
    - x_test:测试数据
    - y_test:测试数据标签
    """
    xt = []
    yt = []
    for b in range(1, 6):  # 读取训练集数据
        filename = os.path.join(cifar10_path, 'data_batch_%d' % (b,))  # 遍历读取data_batch_1到5
        [x, y] = __load_cifar_batch(filename)
        xt.append(x)
        yt.append(y)
    x_train = np.concatenate(xt)  # xt，yt 内为5个(10000, 3, 32, 32)，将其合并成(50000, 3, 32, 32)
    y_train = np.concatenate(yt)
    del x, y
    x_test, y_test = __load_cifar_batch(os.path.join(cifar10_path, 'test_batch'))  # 读取测试集数据
    return [x_train, y_train, x_test, y_test]


def get_cifar_data(num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.

    Inputs
    ------
    - num_training:想要训练的数量
    - num_validation:想要验证的数量
    - num_test:想要测试的数量
    - subtract_mean:子样本是否进行平均

    Returns
    -------

    """
    # Load the raw CIFAR-10 data
    cifar10_directory = 'F:/cs231n/assignment1/cs231n/dataset/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_cifar10(cifar10_directory)

    # 选出子集用于验证、训练、测试
    mask = list(range(num_training, num_training + num_validation))
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    x_test = x_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    # 0值中心化
    if subtract_mean:
        mean_image = np.mean(x_train, axis=0)
        x_train -= mean_image
        x_val -= mean_image
        x_test -= mean_image

    # 转换轴使得通道数位于前面
    x_train = x_train.transpose(0, 3, 1, 2).copy()
    x_val = x_val.transpose(0, 3, 1, 2).copy()
    x_test = x_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        'x_train': x_train, 'y_train': y_train,
        'x_val': x_val, 'y_val': y_val,
        'x_test': x_test, 'y_test': y_test,
    }


def __combine_channel(images):
    """
    将图片数据的三通道合并为彩色图像

    Inputs
    ------
    - images:图片数据 格式(3, 32, 32)

    Returns
    -------
    - img:合并后的彩色图像 PIL格式
    """
    img0 = images[0]
    img1 = images[1]
    img2 = images[2]
    r_channel = Image.fromarray(img0)
    g_channel = Image.fromarray(img1)
    b_channel = Image.fromarray(img2)
    img = Image.merge("RGB", (r_channel, g_channel, b_channel))
    return img


def __show_cifar10(data, label):
    """
    查看数据集中的样本

    Inputs
    ------
    - data:图片数据 格式(50000,3, 32, 32)
    - label:图片对应标签
    """
    classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 类别列表
    num_classes = len(classes)  # 类别数目
    samples_per_class = 7  # 每个类别采样个数
    for y, cls in enumerate(classes):  # 对列表的元素位置和元素进行循环，y表示元素位置（0,num_class），cls元素本身‘plane‘等
        idxs = np.flatnonzero(label == y)  # 找出标签中y类的位置
        idxs = np.random.choice(idxs, samples_per_class, replace=False)  # 从中选出我们所需的7个样本
        for i, idx in enumerate(idxs):  # 对所选的样本的位置和样本所对应的图片在训练集中的位置进行循环
            plt_idx = i * num_classes + y + 1  # 在子图中所占位置的计算
            plt.subplot(samples_per_class, num_classes, plt_idx)  # 说明要画的子图的编号，用法和matlib中subplot相同
            img = __combine_channel(data[idx])  # 将图片数据的三通道合并为彩色图像
            plt.imshow(img)  # 画图
            plt.axis('off')  # 关闭坐标轴
            if i == 0:  # 首行显示类别名
                plt.title(cls)
    plt.show()


def __save_cifar10(data):
    """
    保存数据集中图片

    Inputs
    ------
    - data:图片数据 格式(50000, 3, 32, 32)
    """
    print("正在保存图片:")
    for i in range(data.shape[0]):  # 获取数据中图片的总数
        imgs = data[i - 1]  # 获取单张图片的三通道数据，格式(3, 32, 32)
        if i < 100:  # 只循环100张图片,这句注释掉可以便利出所有的图片,图片较多,可能要一定的时间
            img = __combine_channel(imgs)  # 将图片数据的三通道合并为彩色图像
            img_name = "imgs" + str(i)
            img.save("F:/cs231n/assignment1/dataset/images/" + img_name + ".png")  # 文件夹下是RGB融合后的图像,路径不存在时会报错
            for j in range(imgs.shape[0]):
                img_channel = imgs[j - 1]
                channel_name = "img" + str(i) + "_" + str(j) + ".png"
                print("正在保存图片" + channel_name)
                plimg.imsave("F:/cs231n/assignment1/dataset/image/" + channel_name, img_channel)  # 文件夹下是RGB分离的图像
    print("保存完毕.")


if __name__ == '__main__':
    cifar10_dir = 'F:/cs231n/assignment1/cs231n/dataset/cifar-10-batches-py'
    [X_train, Y_train, X_test, Y_test] = load_cifar10(cifar10_dir)  # 读取cifar-10数据集
    # __show_cifar10(X_train, Y_train)  # 查看数据集中的样本
    # __save_cifar10(X_train)  # 保存数据集中图片
