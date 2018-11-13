# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:TensorFlow.py
@time:2018/11/13 10:00
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

"""Part I. Preparation"""
NUM_TRAIN = 49000
# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True, transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

dtype = torch.float32  # we will be using float throughout this tutorial
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Constant to control how frequently we print train loss
print_every = 100
# print('using device:', device)

"""Part II. Barebones PyTorch"""


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def test_flatten():
    x = torch.arange(12).view(2, 1, 3, 2)
    print('Before flattening:', x)
    print('After flattening:', flatten(x))


def two_layer_fc(x, params):
    """
    A fully-connected neural networks; the architecture is:
    NN is fully connected -> ReLU -> fully connected layer.
    Note that this function only defines the forward pass;
    PyTorch will take care of the backward pass for us.

    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.

    Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).

    Returns:
    - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
      the input data x.
    """
    # first we flatten the image
    x = flatten(x)  # shape: [batch_size, C x H x W]

    w1, w2 = params
    # Forward pass: compute predicted y using operations on Tensors. Since w1 and
    # w2 have requires_grad=True, operations involving these Tensors will cause
    # PyTorch to build a computational graph, allowing automatic computation of
    # gradients. Since we are no longer implementing the backward pass by hand we
    # don't need to keep references to intermediate values.
    # you can also use `.clamp(min=0)`, equivalent to F.relu()
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x


def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros((64, 50), dtype=dtype)  # minibatch size 64, feature dimension 50
    w1 = torch.zeros((50, hidden_layer_size), dtype=dtype)
    w2 = torch.zeros((hidden_layer_size, 10), dtype=dtype)
    scores = two_layer_fc(x, [w1, w2])
    print(scores.size())  # you should see [64, 10]


def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?

    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    conv1 = F.conv2d(x, conv_w1, conv_b1, stride=1, padding=2)
    relu1 = F.relu(conv1)
    conv2 = F.conv2d(relu1, conv_w2, conv_b2, stride=1, padding=1)
    relu2 = F.relu(conv2)
    relu_flatten = flatten(relu2)
    scores = relu_flatten.mm(fc_w) + fc_b
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores


def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]


def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator.
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w


def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)


def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.

    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model

    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))


def train_part2(model_fn, params, learning_rate):
    """
    Train a model on CIFAR-10.

    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD

    Returns: Nothing
    """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device(GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad
                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss=%.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()


"""Part III. PyTorch Module API"""


class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        # forward always defines connectivity
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores


def test_two_layer_fc():
    input_size = 50
    x = torch.zeros((64, input_size), dtype=dtype)  # minibatch size 64, feature dimension 50
    model = TwoLayerFC(input_size, 42, 10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(channel_2*32*32, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def forward(self, x):
        scores = None
        ########################################################################
        # Implement the forward function for a 3-layer ConvNet. you      #
        # should use the layers you defined in __init__ and specify the        #
        # connectivity of those layers in forward()                            #
        ########################################################################
        conv1_out = self.conv1(x)
        relu1_out = self.relu1(conv1_out)
        conv2_out = self.conv2(relu1_out)
        relu2_out = self.relu1(conv2_out)
        scores = self.fc(flatten(relu2_out))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores


def test_three_layer_conv_net():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]


def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))


def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print('Starting epoch %d' % e)
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()
    print('Training finished!')
    check_accuracy_part34(loader_test, model)
    print()


"""Part IV. PyTorch Sequential API"""


# We need to wrap `flatten` function in a module in order to stack it in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


"""Part V. CIFAR-10 open-ended challenge"""


class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # block_1
        self.conv_1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_1_1.weight)
        self.bn_1_1 = nn.BatchNorm2d(num_features=64)
        self.relu_1_1 = nn.ReLU()
        self.drop_1_1 = nn.Dropout2d(p=0.5)

        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_1_2.weight)
        self.bn_1_2 = nn.BatchNorm2d(num_features=64)
        self.relu_1_2 = nn.ReLU()
        self.drop_1_2 = nn.Dropout2d(p=0.5)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block_2
        self.conv_2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_2_1.weight)
        self.bn_2_1 = nn.BatchNorm2d(num_features=128)
        self.relu_2_1 = nn.ReLU()
        self.drop_2_1 = nn.Dropout2d(p=0.5)

        self.conv_2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_2_2.weight)
        self.bn_2_2 = nn.BatchNorm2d(num_features=128)
        self.relu_2_2 = nn.ReLU()
        self.drop_2_2 = nn.Dropout2d(p=0.5)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block_3
        self.conv_3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_3_1.weight)
        self.bn_3_1 = nn.BatchNorm2d(num_features=256)
        self.relu_3_1 = nn.ReLU()
        self.drop_3_1 = nn.Dropout2d(p=0.5)

        self.conv_3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_3_2.weight)
        self.bn_3_2 = nn.BatchNorm2d(num_features=256)
        self.relu_3_2 = nn.ReLU()
        self.drop_3_2 = nn.Dropout2d(p=0.5)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fc
        self.fc = nn.Linear(256*4*4, 10)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        # block_1
        conv_1_1_out = self.conv_1_1(x)
        bn_1_1_out = self.bn_1_1(conv_1_1_out)
        relu_1_1_out = self.relu_1_1(bn_1_1_out)
        drop_1_1_out = self.drop_1_1(relu_1_1_out)

        conv_1_2_out = self.conv_1_2(drop_1_1_out)
        bn_1_2_out = self.bn_1_2(conv_1_2_out)
        relu_1_2_out = self.relu_1_2(bn_1_2_out)
        drop_1_2_out = self.drop_1_2(relu_1_2_out)
        pool_1_out = self.pool_1(drop_1_2_out)

        # block_2
        conv_2_1_out = self.conv_2_1(pool_1_out)
        bn_2_1_out = self.bn_2_1(conv_2_1_out)
        relu_2_1_out = self.relu_2_1(bn_2_1_out)
        drop_2_1_out = self.drop_2_1(relu_2_1_out)

        conv_2_2_out = self.conv_2_2(drop_2_1_out)
        bn_2_2_out = self.bn_2_2(conv_2_2_out)
        relu_2_2_out = self.relu_2_2(bn_2_2_out)
        drop_2_2_out = self.drop_2_2(relu_2_2_out)
        pool_2_out = self.pool_2(drop_2_2_out)

        # block_3
        conv_3_1_out = self.conv_3_1(pool_2_out)
        bn_3_1_out = self.bn_3_1(conv_3_1_out)
        relu_3_1_out = self.relu_3_1(bn_3_1_out)
        drop_3_1_out = self.drop_3_1(relu_3_1_out)

        conv_3_2_out = self.conv_3_2(drop_3_1_out)
        bn_3_2_out = self.bn_3_2(conv_3_2_out)
        relu_3_2_out = self.relu_3_2(bn_3_2_out)
        drop_3_2_out = self.drop_3_2(relu_3_2_out)
        pool_3_out = self.pool_3(drop_3_2_out)

        # fc
        scores = self.fc(flatten(pool_3_out))

        return scores


if __name__ == '__main__':
    """Part II. Barebones PyTorch"""
    # test_flatten()
    # two_layer_fc_test()
    # three_layer_convnet_test()

    # create a weight of shape [3 x 5]
    # you should see the type `torch.cuda.FloatTensor` if you use GPU. Otherwise it should be `torch.FloatTensor`
    # random_weight((3, 5))

    # Train a Two-Layer Network
    # hidden_layer_size = 4000
    # learning_rate = 1e-2
    # w1 = random_weight((3 * 32 * 32, hidden_layer_size))
    # w2 = random_weight((hidden_layer_size, 10))
    # train_part2(two_layer_fc, [w1, w2], learning_rate)

    # Training a ConvNet
    # learning_rate = 3e-3
    # channel_1 = 32
    # channel_2 = 16
    # conv_w1 = random_weight((channel_1, 3, 5, 5))
    # conv_b1 = zero_weight((channel_1,))
    # conv_w2 = random_weight((channel_2, channel_1, 3, 3))
    # conv_b2 = zero_weight((channel_2,))
    # fc_w = random_weight((channel_2 * 32 * 32, 10))
    # fc_b = zero_weight((10,))
    # params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    # train_part2(three_layer_convnet, params, learning_rate)

    """Part III. PyTorch Module API"""
    # test_two_layer_fc()
    # test_three_layer_conv_net()

    # Train a Two-Layer Network
    # hidden_layer_size = 4000
    # learning_rate = 1e-2
    # model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # train_part34(model, optimizer)

    # Train a Three-Layer ConvNet
    # learning_rate = 3e-3
    # channel_1 = 32
    # channel_2 = 16
    # model = ThreeLayerConvNet(in_channel=3, channel_1=channel_1, channel_2=channel_2, num_classes=10)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # train_part34(model, optimizer)

    """Part IV. PyTorch Sequential API"""
    # Sequential API: Two-Layer Network
    # hidden_layer_size = 4000
    # learning_rate = 1e-2
    # model = nn.Sequential(
    #     Flatten(),
    #     nn.Linear(3 * 32 * 32, hidden_layer_size),
    #     nn.ReLU(),
    #     nn.Linear(hidden_layer_size, 10),
    # )
    # # you can use Nesterov momentum in optim.SGD
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # train_part34(model, optimizer)

    # Sequential API: Three-Layer ConvNet
    # channel_1 = 32
    # channel_2 = 16
    # learning_rate = 1e-2
    # model = nn.Sequential(
    #     nn.Conv2d(3, channel_1, kernel_size=5, stride=1, padding=2),
    #     nn.ReLU(),
    #     nn.Conv2d(channel_1, channel_2, kernel_size=3, stride=1, padding=1),
    #     nn.ReLU(),
    #     Flatten(),
    #     nn.Linear(channel_2*32*32, 10),
    # )
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # train_part34(model, optimizer)

    """Part V. CIFAR-10 open-ended challenge"""
    model = MyConvNet()

    # 使用weight_decay 就代表加入了 L2 regularization, 故在训练计算loss时，不用显示的计算L2 reg loss
    # optimizer 会在更新参数的时候计算，但若要加L1损失，则需要自己写，样例如下：
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in model.parameters():
    #     reg_loss += l1_crit(param)
    #
    # factor = 0.0005
    # loss += factor * reg_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    print_every = 700
    # You should get at least 70% accuracy
    train_part34(model, optimizer, epochs=10)
