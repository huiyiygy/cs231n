# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:RNN_Captioning.py
@time:2018/11/30 16:15
"""
import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def check_vanilla_rnn_step_forward():
    """You should see errors on the order of e-8 or less."""
    N, D, H = 3, 10, 4

    x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
    prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
    Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
    Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
    b = np.linspace(-0.2, 0.4, num=H)

    next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
    expected_next_h = np.asarray([
        [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
        [0.66854692, 0.79562378, 0.87755553, 0.92795967],
        [0.97934501, 0.99144213, 0.99646691, 0.99854353]])
    print('next_h error:', rel_error(expected_next_h, next_h))


def check_vanilla_rnn_step_backward():
    np.random.seed(231)
    N, D, H = 4, 5, 6
    x = np.random.randn(N, D)
    h = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)

    next_h, cache = rnn_step_forward(x, h, Wx, Wh, b)

    dnext_h = np.random.randn(*next_h.shape)

    fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fh = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
    dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
    db_num = eval_numerical_gradient_array(fb, b, dnext_h)

    dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)

    print('dx error: ', rel_error(dx_num, dx))
    print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
    print('dWx error: ', rel_error(dWx_num, dWx))
    print('dWh error: ', rel_error(dWh_num, dWh))
    print('db error: ', rel_error(db_num, db))


def check_vanilla_rnn_forward():
    """
    processes an entire sequence of data, You should see errors on the order of  e-7 or less.
    """
    N, T, D, H = 2, 3, 4, 5

    x = np.linspace(-0.1, 0.3, num=N * T * D).reshape(N, T, D)
    h0 = np.linspace(-0.3, 0.1, num=N * H).reshape(N, H)
    Wx = np.linspace(-0.2, 0.4, num=D * H).reshape(D, H)
    Wh = np.linspace(-0.4, 0.1, num=H * H).reshape(H, H)
    b = np.linspace(-0.7, 0.1, num=H)

    h, _ = rnn_forward(x, h0, Wx, Wh, b)
    expected_h = np.asarray([
        [
            [-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
            [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
            [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525],
        ],
        [
            [-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
            [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
            [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043]]])
    print('h error: ', rel_error(expected_h, h))


def check_vanilla_rnn_backward():
    """You should see errors on the order of e-6 or less."""
    np.random.seed(231)

    N, D, T, H = 2, 3, 10, 5

    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)

    out, cache = rnn_forward(x, h0, Wx, Wh, b)

    dout = np.random.randn(*out.shape)

    dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)

    fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
    fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
    fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
    fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
    fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
    db_num = eval_numerical_gradient_array(fb, b, dout)

    print('dx error: ', rel_error(dx_num, dx))
    print('dh0 error: ', rel_error(dh0_num, dh0))
    print('dWx error: ', rel_error(dWx_num, dWx))
    print('dWh error: ', rel_error(dWh_num, dWh))
    print('db error: ', rel_error(db_num, db))


def check_word_embedding_forward():
    """
    In deep learning systems, we commonly represent words using vectors. Each word of the
    vocabulary will be associated with a vector, and these vectors will be learned jointly
    with the rest of the system.
    You should see an error on the order of e-8 or less
    """
    N, T, V, D = 2, 4, 5, 3

    x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
    W = np.linspace(0, 1, num=V * D).reshape(V, D)

    out, _ = word_embedding_forward(x, W)
    expected_out = np.asarray([
        [[0., 0.07142857, 0.14285714],
         [0.64285714, 0.71428571, 0.78571429],
         [0.21428571, 0.28571429, 0.35714286],
         [0.42857143, 0.5, 0.57142857]],
        [[0.42857143, 0.5, 0.57142857],
         [0.21428571, 0.28571429, 0.35714286],
         [0., 0.07142857, 0.14285714],
         [0.64285714, 0.71428571, 0.78571429]]])

    print('out error: ', rel_error(expected_out, out))


def check_word_embedding_backward():
    """You should see an error on the order of e-11 or less."""
    np.random.seed(231)

    N, T, V, D = 50, 3, 5, 6
    x = np.random.randint(V, size=(N, T))
    W = np.random.randn(V, D)

    out, cache = word_embedding_forward(x, W)
    dout = np.random.randn(*out.shape)
    dW = word_embedding_backward(dout, cache)

    f = lambda W: word_embedding_forward(x, W)[0]
    dW_num = eval_numerical_gradient_array(f, W, dout)

    print('dW error: ', rel_error(dW, dW_num))


def check_temporal_affine_layer():
    """
    At every timestep we use an affine function to transform the RNN hidden vector at
    that timestep into scores for each word in the vocabulary. Because this is very
    similar to the affine layer that you implemented in assignment 2, we have provided
    this function for you in the temporal_affine_forward and temporal_affine_backward
    functions in the file cs231n/rnn_layers.py. Run the following to perform numeric
    gradient checking on the implementation. You should see errors on the order of e-9 or less.
    """
    np.random.seed(231)

    # Gradient check for temporal affine layer
    N, T, D, M = 2, 3, 4, 5
    x = np.random.randn(N, T, D)
    w = np.random.randn(D, M)
    b = np.random.randn(M)

    out, cache = temporal_affine_forward(x, w, b)

    dout = np.random.randn(*out.shape)

    fx = lambda x: temporal_affine_forward(x, w, b)[0]
    fw = lambda w: temporal_affine_forward(x, w, b)[0]
    fb = lambda b: temporal_affine_forward(x, w, b)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    dw_num = eval_numerical_gradient_array(fw, w, dout)
    db_num = eval_numerical_gradient_array(fb, b, dout)

    dx, dw, db = temporal_affine_backward(dout, cache)

    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))


def check_temporal_sotfmax_loss():
    """
    In an RNN language model, at every timestep we produce a score for each word in the
    vocabulary. We know the ground-truth word at each timestep, so we use a softmax loss
    function to compute loss and gradient at each timestep. We sum the losses over time
    and average them over the minibatch.

    However there is one wrinkle: since we operate over minibatches and different captions
    may have different lengths, we append <NULL> tokens to the end of each caption so
    they all have the same length. We don't want these <NULL> tokens to count toward the
    loss or gradient, so in addition to scores and ground-truth labels our loss function
    also accepts a mask array that tells it which elements of the scores count towards
    the loss.

    Since this is very similar to the softmax loss function you implemented in assignment 1,
    we have implemented this loss function for you; look at the temporal_softmax_loss
    function in the file cs231n/rnn_layers.py.

    Run the following cell to sanity check the loss and perform numeric gradient checking
    on the function. You should see an error for dx on the order of e-7 or less.
    """
    N, T, V = 100, 1, 10

    def check_loss(N, T, V, p):
        x = 0.001 * np.random.randn(N, T, V)
        y = np.random.randint(V, size=(N, T))
        mask = np.random.rand(N, T) <= p
        print(temporal_softmax_loss(x, y, mask)[0])

    check_loss(100, 1, 10, 1.0)  # Should be about 2.3
    check_loss(100, 10, 10, 1.0)  # Should be about 23
    check_loss(5000, 10, 10, 0.1)  # Should be about 2.3

    # Gradient check for temporal softmax loss
    N, T, V = 7, 8, 9

    x = np.random.randn(N, T, V)
    y = np.random.randint(V, size=(N, T))
    mask = (np.random.rand(N, T) > 0.5)

    loss, dx = temporal_softmax_loss(x, y, mask, verbose=False)
    dx_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], x, verbose=False)

    print('dx error: ', rel_error(dx, dx_num))


if __name__ == "__main__":
    # Load COCO data from disk,this return a dictionary. We'll work with dimensionality-reduced
    # feature for this notebook, but feel free to experiment with the original features by
    # changing the flag below.
    # data = load_coco_data(pca_features=True)
    # # Print out all the keys and values from the data dictionary
    # for k, v in data.items():
    #     if type(v) == np.ndarray:
    #         print(k, type(v), v.shape, v.dtype)
    #     else:
    #         print(k, type(v), len(v))
    #
    # # Sample a minibatch and show the images and captions
    # batch_size = 3
    # captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
    # for i, (caption, url) in enumerate(zip(captions, urls)):
    #     plt.imshow(image_from_url(url))
    #     plt.axis('off')
    #     caption_str = decode_captions(caption, data['idx_to_word'])
    #     plt.title(caption_str)
    #     plt.show()
    check_temporal_affine_layer()
