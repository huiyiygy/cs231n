# -*- coding:utf-8 -*-

"""
@function:
@author:HuiYi or 会意
@file:two_layer_net_test.py
@time:2018/06/08 16:54
"""
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_cifar10
from cs231n.vis_utils import visualize_grid
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.gradient_check import eval_numerical_gradient


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
    # 将图像数据转置成二维的
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    # 预处理减去图片的平均值
    mean_image = np.mean(X_train, axis=0)  # from (49000, 3072) to (1, 3072)
    X_train -= mean_image
    X_test -= mean_image
    X_val -= mean_image
    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def init_toy_model(input_size, hidden_size, num_classes):
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


def init_toy_data(num_inputs, input_size):
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


def toymodel_net_test():
    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5

    net = init_toy_model(input_size, hidden_size, num_classes)
    X, y = init_toy_data(num_inputs, input_size)

    # 前向传播：计算分值
    scores = net.loss(X)
    print('Your scores:')
    print(scores)
    print()
    print('correct scores:')
    correct_scores = np.asarray([
        [-0.81233741, -1.27654624, -0.70335995],
        [-0.17129677, -1.18803311, -0.47310444],
        [-0.51590475, -1.01354314, -0.8504215],
        [-0.15419291, -0.48629638, -0.52901952],
        [-0.00618733, -0.12435261, -0.15226949]])
    print(correct_scores)
    print()
    # The difference should be very small. We get < 1e-7
    print('Difference between your scores and correct scores:')
    print(np.sum(np.abs(scores - correct_scores)))

    # 前向传播：计算损失值
    loss, _ = net.loss(X, y, reg=0.05)
    correct_loss = 1.30378789133
    # should be very small, we get < 1e-12
    print('Difference between your loss and correct loss:')
    print(np.sum(np.abs(loss - correct_loss)))

    # 反向传播
    loss, grads = net.loss(X, y, reg=0.05)

    # these should all be less than 1e-8 or so
    for param_name in grads:
        f = lambda W: net.loss(X, y, reg=0.05)[0]
        param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

    # 训练
    stats = net.train(X, y, X, y,
                      learning_rate=1e-1, reg=5e-6,
                      num_iters=100, verbose=False)

    print('Final training loss: ', stats['loss_history'][-1])

    # plot the loss history
    plt.plot(stats['loss_history'])
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


if __name__ == '__main__':
    # 小模型测试
    # toymodel_net_test()

    X_train, y_train, X_test, y_test, X_val, y_val = get_data()
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val, num_iters=10000, batch_size=200,
                      learning_rate=1e-3, learning_rate_decay=0.95, reg=0.25, verbose=True)

    # Predict on the validation set
    val_acc = np.mean(net.predict(X_val) == y_val)
    print('Validation accuracy: ', val_acc)

    # Predict on the test set
    test_acc = np.mean(net.predict(X_test) == y_test)
    print('Test accuracy: ', test_acc)

    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.tight_layout()  # 调整子图间距
    plt.show()

    # 可视化网络权重
    show_net_weights(net)
