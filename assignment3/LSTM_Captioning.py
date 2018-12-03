# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: LSTM_Captioning.py
@time: 2018/12/3 14:50
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


def check_lstm_step_forward():
    """You should see errors on the order of e-8 or less."""
    N, D, H = 3, 4, 5
    x = np.linspace(-0.4, 1.2, num=N * D).reshape(N, D)
    prev_h = np.linspace(-0.3, 0.7, num=N * H).reshape(N, H)
    prev_c = np.linspace(-0.4, 0.9, num=N * H).reshape(N, H)
    Wx = np.linspace(-2.1, 1.3, num=4 * D * H).reshape(D, 4 * H)
    Wh = np.linspace(-0.7, 2.2, num=4 * H * H).reshape(H, 4 * H)
    b = np.linspace(0.3, 0.7, num=4 * H)

    next_h, next_c, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)

    expected_next_h = np.asarray([
        [0.24635157, 0.28610883, 0.32240467, 0.35525807, 0.38474904],
        [0.49223563, 0.55611431, 0.61507696, 0.66844003, 0.7159181],
        [0.56735664, 0.66310127, 0.74419266, 0.80889665, 0.858299]])
    expected_next_c = np.asarray([
        [0.32986176, 0.39145139, 0.451556, 0.51014116, 0.56717407],
        [0.66382255, 0.76674007, 0.87195994, 0.97902709, 1.08751345],
        [0.74192008, 0.90592151, 1.07717006, 1.25120233, 1.42395676]])

    print('next_h error: ', rel_error(expected_next_h, next_h))
    print('next_c error: ', rel_error(expected_next_c, next_c))


def check_lstm_step_backward():
    """You should see errors on the order of e-7 or less"""
    np.random.seed(231)

    N, D, H = 4, 5, 6
    x = np.random.randn(N, D)
    prev_h = np.random.randn(N, H)
    prev_c = np.random.randn(N, H)
    Wx = np.random.randn(D, 4 * H)
    Wh = np.random.randn(H, 4 * H)
    b = np.random.randn(4 * H)

    next_h, next_c, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)

    dnext_h = np.random.randn(*next_h.shape)
    dnext_c = np.random.randn(*next_c.shape)

    fx_h = lambda x: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
    fh_h = lambda h: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
    fc_h = lambda c: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
    fWx_h = lambda Wx: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
    fWh_h = lambda Wh: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
    fb_h = lambda b: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]

    fx_c = lambda x: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
    fh_c = lambda h: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
    fc_c = lambda c: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
    fWx_c = lambda Wx: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
    fWh_c = lambda Wh: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
    fb_c = lambda b: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]

    num_grad = eval_numerical_gradient_array

    dx_num = num_grad(fx_h, x, dnext_h) + num_grad(fx_c, x, dnext_c)
    dh_num = num_grad(fh_h, prev_h, dnext_h) + num_grad(fh_c, prev_h, dnext_c)
    dc_num = num_grad(fc_h, prev_c, dnext_h) + num_grad(fc_c, prev_c, dnext_c)
    dWx_num = num_grad(fWx_h, Wx, dnext_h) + num_grad(fWx_c, Wx, dnext_c)
    dWh_num = num_grad(fWh_h, Wh, dnext_h) + num_grad(fWh_c, Wh, dnext_c)
    db_num = num_grad(fb_h, b, dnext_h) + num_grad(fb_c, b, dnext_c)

    dx, dh, dc, dWx, dWh, db = lstm_step_backward(dnext_h, dnext_c, cache)

    print('dx error: ', rel_error(dx_num, dx))
    print('dh error: ', rel_error(dh_num, dh))
    print('dc error: ', rel_error(dc_num, dc))
    print('dWx error: ', rel_error(dWx_num, dWx))
    print('dWh error: ', rel_error(dWh_num, dWh))
    print('db error: ', rel_error(db_num, db))


if __name__ == "__main__":
    # data = load_coco_data(pca_features=True)
    # Print out all the keys and values from the data dictionary
    # for k, v in data.items():
    #     if type(v) == np.ndarray:
    #         print(k, type(v), v.shape, v.dtype)
    #     else:
    #         print(k, type(v), len(v))
    check_lstm_step_backward()
