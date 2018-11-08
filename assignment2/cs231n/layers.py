# -*- coding:utf-8 -*-
"""
@author:HuiYi or 会意
@file:layers.py
@time:2018/09/01 16:42
"""
import numpy as np


def affine_forward(x, w, b):
    """
    Compute the forward pass for an affine (fully-connected) layer.

    The input x has shape(N, d_1, ..., d_k) and contains a minibatch of N examples, where each
    example x[i] has shape(d_1, ..., d_k). We will reshape each input into a vector of dimension
     D = d_1 * ... * d_k, and then transform it to an output vector of dimension M.

    Inputs:
    -------
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Return:
    -------
    - out: output, of shape(N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    #  Implement the affine forward pass. Store the result in out. You        #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_temp = np.reshape(x, (x.shape[0], -1))
    out = x_temp.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    -------
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns:
    -------
    a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    #  Implement the affine backward pass.                                    #
    ###########################################################################
    dx = dout.dot(w.T)
    dx = np.reshape(dx, x.shape)
    x_temp = np.reshape(x, (x.shape[0], -1))
    dw = x_temp.T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    ------
    - x: Inputs, of any shape

    Returns:
    --------
    a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # Implement the ReLU forward pass.                                        #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    ------
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    --------
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # Implement the ReLU backward pass.                                       #
    ###########################################################################
    dx = dout
    dx[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    在训练期间，样本均值和（未校正的）样本方差是根据小批量统计计算的，用于标准化输入数据。
    在训练期间，我们还保持每个特征的均值和方差的指数衰减运行平均值，并且这些平均值用于在测试时标准化数据

    在每个时间步，我们使用基于动量参数的指数衰减更新均值和方差的运行平均值：

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    请注意，batch normalization论文提出了不同的测试时间行为：它们使用大量训练图像而不是使用运行平均值
    来计算每个要素的样本均值和方差。 对于此实现，我们选择使用运行平均值，因为它们不需要额外的估计步骤;
    批量标准化的torch7实现也使用运行平均值。

    Input:
    -------
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features

    Returns:
    --------
    a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    _, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # Implement the training-time forward pass for batch norm.            #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta

        cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # Implement the test-time forward pass for batch normalization.       #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    优化方法见下方batchnorm_backward_alt 函数

    For this implementation, you should write out a computation graph for batch normalization
    on paper and propagate gradients backward through intermediate nodes(中间节点).

    Input:
    ------
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns:
    --------
    a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # Implement the backward pass for batch normalization. Store the          #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, sample_mean, sample_var, x_hat, eps, gamma, beta = cache
    N = x.shape[0]

    dx_1 = gamma * dout
    dx_2_b = np.sum((x - sample_mean) * dx_1, axis=0)
    dx_2_a = ((sample_var + eps) ** -0.5) * dx_1
    dx_3_b = (-0.5) * ((sample_var + eps) ** -1.5) * dx_2_b
    dx_4_b = dx_3_b * 1
    dx_5_b = np.ones_like(x) / N * dx_4_b
    dx_6_b = 2 * (x - sample_mean) * dx_5_b
    dx_7_a = dx_6_b * 1 + dx_2_a * 1
    dx_7_b = dx_6_b * 1 + dx_2_a * 1
    dx_8_b = -1 * np.sum(dx_7_b, axis=0)
    dx_9_b = np.ones_like(x) / N * dx_8_b
    dx_10 = dx_9_b + dx_7_a

    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dx_10
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch normalizaton
    backward pass on paper and simplify as much as possible. You should be able to derive a
    simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # Implement the backward pass for batch normalization. Store the          #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, sample_mean, sample_var, x_hat, eps, gamma, beta = cache
    N = x.shape[0]
    dgamma = np.sum(dout * x_hat, axis=0)  # 第5行公式
    dbeta = np.sum(dout, axis=0)  # 第6行公式
    dx_hat = dout * gamma  # 第1行公式
    dx_hat_numerator = dx_hat / np.sqrt(sample_var + eps)  # 第3行第1项（未负求和）
    dx_hat_denominator = np.sum(dx_hat * (x - sample_mean), axis=0)  # 第2行前半部分
    dx_1 = dx_hat_numerator  # 第4行第1项
    dvar = -0.5 * ((sample_var + eps) ** (-1.5)) * dx_hat_denominator  # 第2行公式
    # Note sample_var is also a function of mean
    dmean = -1.0 * np.sum(dx_hat_numerator, axis=0) + dvar * np.mean(-2.0 * (x - sample_mean), axis=0)  # 第3行公式（部分）
    dx_var = dvar * 2.0 / N * (x - sample_mean)  # 第4行第2项
    dx_mean = dmean * 1.0 / N  # 第4行第3项
    # with shape(D,), no trouble with broadcast
    dx = dx_1 + dx_var + dx_mean
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    ------
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # Implement the training-time forward pass for layer norm.                #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)

    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_hat + beta

    cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    -------
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns:
    --------
    a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # Implement the backward pass for layer norm.                             #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, sample_mean, sample_var, x_hat, eps, gamma, beta = cache
    N = x.shape[0]
    dgamma = np.sum(dout * x_hat, axis=0)  # 第5行公式
    dbeta = np.sum(dout, axis=0)  # 第6行公式
    dx_hat = dout * gamma  # 第1行公式
    dx_hat_numerator = dx_hat / np.sqrt(sample_var +eps)  # 第3行第1项（未负求和）
    dx_hat_denominator = np.sum(dx_hat * (x - sample_mean), axis=0)  # 第2行前半部分
    dx_1 = dx_hat_numerator  # 第4行第1项
    dvar = -0.5 * ((sample_var + eps) ** (-1.5)) * dx_hat_denominator  # 第2行公式
    # Note sample_var is also a function of mean
    dmean = -1.0 * np.sum(dx_hat_numerator, axis=0) + dvar * np.mean(-2.0 * (x - sample_mean), axis=0)  # 第3行公式（部分）
    dx_var = dvar * 2.0 / N * (x - sample_mean)  # 第4行第2项
    dx_mean = dmean * 1.0 / N  # 第4行第3项
    # with shape(D,), no trouble with broadcast
    dx = dx_1 + dx_var + dx_mean
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    -------
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Returns:
    --------
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # Implement training phase forward pass for inverted dropout.         #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # np.random.rand返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # Implement the test phase forward pass for inverted dropout.         #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    -------
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # Implement training phase backward pass for inverted dropout         #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    ------
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns:
    --------
    a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # Implement the convolutional forward pass.                               #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    out = np.zeros((N, F, H_out, W_out), dtype=np.float64)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_mask = x_pad[:, :, (i * stride):(i * stride + HH), (j * stride):(j * stride + WW)]
            for k in range(F):
                out[:, k, i, j] = np.sum(x_pad_mask * w[k, :, :, :], axis=(1, 2, 3))
    out = out + b[None, :, None, None]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    -------
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns:
    --------
    a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)

    db = np.sum(dout, axis=(0, 2, 3))
    for i in range(H_out):
        for j in range(W_out):
            x_pad_mask = x_pad[:, :, (i * stride):(i * stride + HH), (j * stride):(j * stride + WW)]
            for k in range(F):
                dw[k, :, :, :] += np.sum(x_pad_mask * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N):
                dx_pad[n, :, (i * stride):(i * stride + HH), (j * stride):(j * stride + WW)] += np.sum(w * (dout[n, :, i, j])[:, None, None, None], axis=0)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    -------
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns:
    --------
    a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # Implement the max-pooling forward pass                                  #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, (i*stride):(i*stride+pool_height), (j*stride):(j*stride+pool_width)]
            out[:, :, i, j] = np.max(x_mask, axis=(2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    -------
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    --------
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # Implement the max-pooling backward pass                                 #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    _, _, H_out, W_out = dout.shape
    dx = np.zeros_like(x)
    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, (i*stride):(i*stride+pool_height), (j*stride):(j*stride+pool_width)]
            masked_x_mask = np.max(x_mask, axis=(2, 3), keepdims=True)
            # flags: only the max value is True, others are False
            flags = masked_x_mask == x_mask
            dx[:, :, (i*stride):(i*stride+pool_height), (j*stride):(j*stride+pool_width)] = flags * (dout[:, :, i, j])[:, :, None, None]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    -------
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns:
    --------
    a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ###########################################################################
    # Implement the forward pass for spatial batch normalization.             #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    temp_out, cache = batchnorm_forward(x.transpose(0, 2, 3, 1).reshape((N*H*W, C)), gamma, beta, bn_param)
    out = temp_out.reshape((N, H, W, C)).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    -------
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns:
    --------
    a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # Implement the backward pass for spatial batch normalization.            #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0, 2, 3, 1).reshape((N*H*W, C)), cache)
    dx = dx_temp.reshape((N, H, W, C)).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization
    and layer normalization.

    Inputs:
    -------
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1,C,1,1)
    - beta: Shift parameter, of shape (1,C,1,1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns:
    --------
    a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps', 1e-5)
    ###########################################################################
    # Implement the forward pass for spatial group normalization.             #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape
    x_group = np.reshape(x, [N, G, C // G, H, W])  # 按G将C分组
    mean = np.mean(x_group, axis=(2, 3, 4), keepdims=True)  # 均值
    var = np.var(x_group, axis=(2, 3, 4), keepdims=True)  # 方差
    x_group_norm = (x_group - mean) / np.sqrt(var + eps)  # 归一化
    x_norm = np.reshape(x_group_norm, [N, C, H, W])  # 还原维度
    out = gamma * x_norm + beta
    cache = (x_group, mean, var, x_norm, G, eps, gamma, beta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    -------
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns:
    --------
    a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # Implement the backward pass for spatial group normalization.            #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    x_group, mean, var, x_norm, G, eps, gamma, beta = cache
    N, C, H, W = dout.shape
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)

    # dx_group_norm
    dx_norm = dout * gamma
    dx_group_norm = np.reshape(dx_norm, [N, G, C // G, H, W])
    # dvar
    dvar = -0.5 * ((var + eps) ** (-1.5)) * np.sum(dx_group_norm * (x_group - mean), axis=(2, 3, 4), keepdims=True)
    # dmean
    N_GROUP = C // G*H*W
    dmean1 = np.sum(dx_group_norm * -1.0 / np.sqrt(var + eps), axis=(2, 3, 4), keepdims=True)
    dmean2_var = dvar * -2.0 / N_GROUP * np.sum(x_group - mean, axis=(2, 3, 4), keepdims=True)
    dmean = dmean1 + dmean2_var
    # dx_group
    dx_group1 = dx_group_norm * 1.0 / np.sqrt(var + eps)
    dx_group_var = dvar * 2.0 / N_GROUP * (x_group - mean)
    dx_group_mean = dmean * 1.0 / N_GROUP
    dx_group = dx_group1 + dx_group_var + dx_group_mean
    # dx
    dx = np.reshape(dx_group, [N, C, H, W])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    -------
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
    -------
    a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    -------
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
    --------
    a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
