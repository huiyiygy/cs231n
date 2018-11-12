# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:TensorFlow.py
@time:2018/11/10 15:04
"""
import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

"""Part I: Preparation"""


def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()  # Return a copy of the array collapsed into one dimension
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        -------
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))


"""Part II: Barebone TensorFlow"""


def flatten(x):
    """
    Inputs:
    -------
    - TensorFlow Tensor of shape (N, D1, ..., DM)
    Returns:
    --------
    - TensorFlow Tensor of shape (N, D1 * ... * DM)
    """
    N = tf.shape(x)[0]
    return tf.reshape(x, (N, -1))


def flatten_test():
    # Clear the current TensorFlow graph.
    tf.reset_default_graph()

    # Stage I: Define the TensorFlow graph describing our computation.
    # In this case the computation is trivial: we just want to flatten
    # a Tensor using the flatten function defined above.

    # Our computation will have a single input, x. We don't know its
    # value yet, so we define a placeholder which will hold the value
    # when the graph is run. We then pass this placeholder Tensor to
    # the flatten function; this gives us a new Tensor which will hold
    # a flattened view of x when the graph is run. The tf.device
    # context manager tells TensorFlow whether to place these Tensors
    # on CPU or GPU.
    with tf.device(device):
        x = tf.placeholder(tf.float32)
        x_flat = flatten(x)

    # At this point we have just built the graph describing our computation,
    # but we haven't actually computed anything yet. If we print x and x_flat
    # we see that they don't hold any data; they are just TensorFlow Tensors
    # representing values that will be computed when the graph is run.
    print('x: ', type(x), x)
    print('x_flat: ', type(x_flat), x_flat)
    print()

    # We need to use a TensorFlow Session object to actually run the graph.
    with tf.Session() as sess:
        # Construct concrete values of the input data x using numpy
        x_np = np.arange(24).reshape((2, 3, 4))
        print('x_np:\n', x_np, '\n')

        # Run our computational graph to compute a concrete output value.
        # The first argument to sess.run tells TensorFlow which Tensor
        # we want it to compute the value of; the feed_dict specifies
        # values to plug into all placeholder nodes in the graph. The
        # resulting value of x_flat is returned from sess.run as a
        # numpy array.
        x_flat_np = sess.run(x_flat, feed_dict={x: x_np})
        print('x_flat_np:\n', x_flat_np, '\n')

        # We can reuse the same graph to perform the same computation
        # with different input data
        x_np = np.arange(12).reshape((2, 3, 2))
        print('x_np:\n', x_np, '\n')
        x_flat_np = sess.run(x_flat, feed_dict={x: x_np})
        print('x_flat_np:\n', x_flat_np)


def two_layer_fc(x, params):
    """
    A fully-connected neural network; the architecture is:
    fully-connected layer -> ReLU -> fully connected layer.
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.

    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.

    Inputs:
    -------
    - x: A TensorFlow Tensor of shape (N, d1, ..., dM) giving a minibatch of input data.
    - params: A list [w1, w2] of TensorFlow Tensors giving weights for the
      network, where w1 has shape (D, H) and w2 has shape (H, C).
    Returns:
    --------
    - scores: A TensorFlow Tensor of shape (N, C) giving classification scores
      for the input data x.
    """
    w1, w2 = params  # Unpack the parameters
    x = flatten(x)  # Flatten the input; now x has shape (N, D)
    h = tf.nn.relu(tf.matmul(x, w1))  # Hidden layer: h has shape (N, H)
    scores = tf.matmul(h, w2)  # Compute scores of shape (N, C)
    return scores


def two_layer_fc_test():
    """
    test two layer fully connected net
    """
    # TensorFlow's default computational graph is essentially a hidden global
    # variable. To avoid adding to this default graph when you rerun this cell,
    # we clear the default graph before constructing the graph we care about.
    tf.reset_default_graph()
    hidden_layer_size = 42
    # Scoping our computational graph setup code under a tf.device context
    # manager lets us tell TensorFlow where we want these Tensors to be placed.
    with tf.device(device):
        # Set up a placehoder for the input of the network, and constant
        # zero Tensors for the network weights. Here we declare w1 and w2
        # using tf.zeros instead of tf.placeholder as we've seen before - this
        # means that the values of w1 and w2 will be stored in the computational
        # graph itself and will persist across multiple runs of the graph; in
        # particular this means that we don't have to pass values for w1 and w2
        # using a feed_dict when we eventually run the graph.
        x = tf.placeholder(tf.float32)
        w1 = tf.zeros((32*32*3, hidden_layer_size))
        w2 = tf.zeros((hidden_layer_size, 10))

        # Call our two_layer_fc function to set up the computational
        # graph for the forward pass of the network.
        scores = two_layer_fc(x, [w1, w2])

    # Use numpy to create some concrete data that we will pass to the
    # computational graph for the x placeholder.
    x_np = np.zeros((64, 32, 32, 3))
    with tf.Session() as sess:
        # The calls to tf.zeros above do not actually instantiate the values
        # for w1 and w2; the following line tells TensorFlow to instantiate
        # the values of all Tensors (like w1 and w2) that live in the graph.
        sess.run(tf.global_variables_initializer())

        # Here we actually run the graph, using the feed_dict to pass the
        # value to bind to the placeholder for x; we ask TensorFlow to compute
        # the value of the scores Tensor, which it returns as a numpy array.
        scores_np = sess.run(scores, feed_dict={x: x_np})
        print(scores_np.shape)


def three_layer_convnet(x, params):
    """
    A three-layer convolutional network with the architecture described above.

    Here you will complete the implementation of the function three_layer_convnet which will perform the forward pass of
    a three-layer convolutional network.  The network should have the following architecture:

    - A convolutional layer (with bias) with channel_1 filters, each with shape KW1 x KH1, and zero-padding of two
    - ReLU nonlinearity
    - A convolutional layer (with bias) with channel_2 filters, each with shape KW2 x KH2, and zero-padding of one
    - ReLU nonlinearity
    - Fully-connected layer with bias, producing scores for C classes.

    HINT: For convolutions: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d; be careful with padding!
    HINT: For biases: https://www.tensorflow.org/performance/xla/broadcasting

    Inputs:
    - x: A TensorFlow Tensor of shape (N, H, W, 3) giving a minibatch of images
    - params: A list of TensorFlow Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: TensorFlow Tensor of shape (KH1, KW1, 3, channel_1) giving weights for the first convolutional layer.
      - conv_b1: TensorFlow Tensor of shape (channel_1,) giving biases for the first convolutional layer.
      - conv_w2: TensorFlow Tensor of shape (KH2, KW2, channel_1, channel_2) giving weights for the second convolutional layer
      - conv_b2: TensorFlow Tensor of shape (channel_2,) giving biases for the second convolutional layer.
      - fc_w: TensorFlow Tensor giving weights for the fully-connected layer.
      - fc_b: TensorFlow Tensor giving biases for the fully-connected layer.
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ############################################################################
    # Implement the forward pass for the three-layer ConvNet.            #
    ############################################################################
    conv1 = tf.nn.conv2d(x, conv_w1, strides=[1, 1, 1, 1], padding='SAME', name='conv1') + conv_b1
    relu1 = tf.nn.relu(conv1)
    conv2 = tf.nn.conv2d(relu1, conv_w2, strides=[1, 1, 1, 1], padding='SAME', name='conv2') + conv_b2
    relu2 = tf.nn.relu(conv2)
    relu2 = tf.layers.flatten(relu2)
    scores = tf.matmul(relu2, fc_w) + fc_b
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
    return scores


def three_layer_convnet_test():
    tf.reset_default_graph()

    with tf.device(device):
        x = tf.placeholder(tf.float32)
        conv_w1 = tf.zeros((5, 5, 3, 6))
        conv_b1 = tf.zeros((6,))
        conv_w2 = tf.zeros((3, 3, 6, 9))
        conv_b2 = tf.zeros((9,))
        fc_w = tf.zeros((32 * 32 * 9, 10))
        fc_b = tf.zeros((10,))
        params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
        scores = three_layer_convnet(x, params)

    # Inputs to convolutional layers are 4-dimensional arrays with shape
    # [batch_size, height, width, channels]
    x_np = np.zeros((64, 32, 32, 3))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores, feed_dict={x: x_np})
        print('scores_np has shape: ', scores_np.shape)


def training_step(scores, y, params, learning_rate):
    """
    Set up the part of the computational graph which makes a training step.

    We now define the `training_step` function which sets up the part of the computational graph that performs a single
    training step. This will take three basic steps:

    1. Compute the loss
    2. Compute the gradient of the loss with respect to all network weights
    3. Make a weight update step using (stochastic) gradient descent.

    Note that the step of updating the weights is itself an operation in the computational graph - the calls to
    `tf.assign_sub` in `training_step` return TensorFlow operations that mutate the weights when they are executed.
    There is an important bit of subtlety here - when we call `sess.run`, TensorFlow does not execute all operations in
    the computational graph; it only executes the minimal subset of the graph necessary to compute the outputs that we
    ask TensorFlow to produce. As a result, naively computing the loss would not cause the weight update operations to
    execute, since the operations needed to compute the loss do not depend on the output of the weight update. To fix
    this problem, we insert a **control dependency** into the graph, adding a duplicate `loss` node to the graph that
    does depend on the outputs of the weight update operations; this is the object that we actually return from the
    `training_step` function. As a result, asking TensorFlow to evaluate the value of the `loss` returned from
    `training_step` will also implicitly update the weights of the network using that minibatch of data.

    We need to use a few new TensorFlow functions to do all of this:
    - For computing the cross-entropy loss we'll use `tf.nn.sparse_softmax_cross_entropy_with_logits`:
        https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    - For averaging the loss across a minibatch of data we'll use `tf.reduce_mean`:
    https://www.tensorflow.org/api_docs/python/tf/reduce_mean
    - For computing gradients of the loss with respect to the weights we'll use `tf.gradients`:
        https://www.tensorflow.org/api_docs/python/tf/gradients
    - We'll mutate the weight values stored in a TensorFlow Tensor using `tf.assign_sub`:
        https://www.tensorflow.org/api_docs/python/tf/assign_sub
    - We'll add a control dependency to the graph using `tf.control_dependencies`:
        https://www.tensorflow.org/api_docs/python/tf/control_dependencies

    Inputs:
    - scores: TensorFlow Tensor of shape (N, C) giving classification scores for
      the model.
    - y: TensorFlow Tensor of shape (N,) giving ground-truth labels for scores;
      y[i] == c means that c is the correct class for scores[i].
    - params: List of TensorFlow Tensors giving the weights of the model
    - learning_rate: Python scalar giving the learning rate to use for gradient
      descent step.

    Returns:
    - loss: A TensorFlow Tensor of shape () (scalar) giving the loss for this
      batch of data; evaluating the loss also performs a gradient descent step
      on params (see above).
    """
    # First compute the loss; the first line gives losses for each example in
    # the minibatch, and the second averages the losses acros the batch
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss = tf.reduce_mean(losses)
    # Compute the gradient of the loss with respect to each parameter of the the
    # network. This is a very magical function call: TensorFlow internally
    # traverses the computational graph starting at loss backward to each element
    # of params, and uses backpropagation to figure out how to compute gradients;
    # it then adds new operations to the computational graph which compute the
    # requested gradients, and returns a list of TensorFlow Tensors that will
    # contain the requested gradients when evaluated.
    grad_params = tf.gradients(loss, params)
    # Make a gradient descent step on all of the model parameters.
    new_weights = []
    for w, grad_w in zip(params, grad_params):
        new_w = tf.assign_sub(w, learning_rate * grad_w)
        new_weights.append(new_w)
    # Insert a control dependency so that evaluting the loss causes a weight
    # update to happen; see the discussion above.
    with tf.control_dependencies(new_weights):
        return tf.identity(loss)


def check_accuracy(sess, dset, x, scores, is_training=None):
    """
    Check accuracy on a classification model.

    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))


def train_part2(model_fn, init_fn, learning_rate):
    """
    Now we set up a basic training loop using low-level TensorFlow operations. We will train the model using stochastic
    gradient descent without momentum. The training_step function sets up the part of the computational graph that
    performs the training step, and the function train_part2 iterates through the training data, making training steps
    on each minibatch, and periodically evaluates accuracy on the validation set.

    Train a model on CIFAR-10.

    Inputs:
    - model_fn: A Python function that performs the forward pass of the model
      using TensorFlow; it should have the following signature:
      scores = model_fn(x, params) where x is a TensorFlow Tensor giving a
      minibatch of image data, params is a list of TensorFlow Tensors holding
      the model weights, and scores is a TensorFlow Tensor of shape (N, C)
      giving scores for all elements of x.
    - init_fn: A Python function that initializes the parameters of the model.
      It should have the signature params = init_fn() where params is a list
      of TensorFlow Tensors holding the (randomly initialized) weights of the
      model.
    - learning_rate: Python float giving the learning rate to use for SGD.
    """
    # First clear the default graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='is_training')
    # Set up the computational graph for performing forward and backward passes, and weight updates.
    with tf.device(device):
        # Set up placeholders for the data and labels
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])
        params = init_fn()  # Initialize the model parameters
        scores = model_fn(x, params)  # Forward pass of the model
        loss = training_step(scores, y, params, learning_rate)

    # Now we actually run the graph many times using the training data
    with tf.Session() as sess:
        # Initialize variables that will live in the graph
        sess.run(tf.global_variables_initializer())
        for t, (x_np, y_np) in enumerate(train_dset):
            # Run the graph on a batch of training data; recall that asking
            # TensorFlow to evaluate loss will cause an SGD step to happen.
            feed_dict = {x: x_np, y: y_np}
            loss_np = sess.run(loss, feed_dict=feed_dict)

            # Periodically print the loss and check accuracy on the val set
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss_np))
                check_accuracy(sess, val_dset, x, scores, is_training)


def kaiming_normal(shape):
    fan_in, fan_out = 0, 0
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]  # np.prod将里面所有的元素相乘
    #  tf.random_normal 返回服从均值为0，方差为1的高斯分布
    return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)


def two_layer_fc_init():
    """
    Initialize the weights of a two-layer network, for use with the
    two_layer_network function defined above.

    Inputs: None

    Returns: A list of:
    - w1: TensorFlow Variable giving the weights for the first layer
    - w2: TensorFlow Variable giving the weights for the second layer
    """
    hidden_layer_size = 4000
    # tf.Variable() A variable maintains state in the graph across calls to `run()`
    w1 = tf.Variable(kaiming_normal((3 * 32 * 32, hidden_layer_size)))
    w2 = tf.Variable(kaiming_normal((hidden_layer_size, 10)))
    return [w1, w2]


def three_layer_convnet_init():
    """
    Initialize the weights of a Three-Layer ConvNet, for use with the
    three_layer_convnet function defined above.

    Inputs: None

    Returns a list containing:
    - conv_w1: TensorFlow Variable giving weights for the first conv layer
    - conv_b1: TensorFlow Variable giving biases for the first conv layer
    - conv_w2: TensorFlow Variable giving weights for the second conv layer
    - conv_b2: TensorFlow Variable giving biases for the second conv layer
    - fc_w: TensorFlow Variable giving weights for the fully-connected layer
    - fc_b: TensorFlow Variable giving biases for the fully-connected layer
    """
    params = None
    ############################################################################
    # Initialize the parameters of the three-layer network.              #
    ############################################################################
    conv_w1 = tf.Variable(kaiming_normal((5, 5, 3, 32)))
    conv_b1 = tf.Variable(tf.zeros((32,)))
    conv_w2 = tf.Variable(kaiming_normal((3, 3, 32, 16)))
    conv_b2 = tf.Variable(tf.zeros((16,)))
    fc_w = tf.Variable(kaiming_normal((32*32*16, 10)))
    fc_b = tf.Variable(tf.zeros((10,)))
    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return params


"""Part III: Keras API"""


class TwoLayerFC(tf.keras.Model):
    """"""
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        initializer = tf.variance_scaling_initializer(scale=2.0)
        self.fc1 = tf.layers.Dense(hidden_size, activation=tf.nn.relu, kernel_initializer=initializer)
        self.fc2 = tf.layers.Dense(num_classes, kernel_initializer=initializer)

    def call(self, x, training=None):
        x = tf.layers.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test_two_layer_fc():
    """ A small unit test to exercise the TwoLayerFC model above. """
    tf.reset_default_graph()
    input_size, hidden_size, num_classes = 50, 42, 10
    # As usual in TensorFlow, we first need to define our computational graph.
    # To this end we first construct a TwoLayerFC object, then use it to construct
    # the scores Tensor.
    model = TwoLayerFC(hidden_size, num_classes)
    with tf.device(device):
        x = tf.zeros((64, input_size))
        scores = model(x)

    # Now that our computational graph has been defined we can run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape)


def two_layer_fc_functional(inputs, hidden_size, num_classes):
    initializer = tf.variance_scaling_initializer(scale=2.0)
    flattened_inputs = tf.layers.flatten(inputs)
    fc1_output = tf.layers.dense(flattened_inputs, hidden_size, activation=tf.nn.relu, kernel_initializer=initializer)
    scores = tf.layers.dense(fc1_output, num_classes, kernel_initializer=initializer)
    return scores


def test_two_layer_fc_functional():
    """ A small unit test to exercise the TwoLayerFC model above. """
    tf.reset_default_graph()
    input_size, hidden_size, num_classes = 50, 42, 10

    # As usual in TensorFlow, we first need to define our computational graph.
    # To this end we first construct a two layer network graph by calling the
    # two_layer_network() function. This function constructs the computation
    # graph and outputs the score tensor.
    with tf.device(device):
        x = tf.zeros((64, input_size))
        scores = two_layer_fc_functional(x, hidden_size, num_classes)

    # Now that our computational graph has been defined we can run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape)


class ThreeLayerConvNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2, num_classes):
        super().__init__()  # tensorflow1.6及以下版本此句会报错，不支持此写法
        ########################################################################
        # Implement the __init__ method for a three-layer ConvNet. You   #
        # should instantiate layer objects to be used in the forward pass.     #
        ########################################################################
        initializer = tf.variance_scaling_initializer(scale=2.0)
        self.conv1 = tf.layers.Conv2D(filters=channel_1, kernel_size=(5, 5), padding='same', strides=[1, 1],
                                      kernel_initializer=initializer, data_format='channels_last',
                                      activation=tf.nn.relu, name='conv1')
        self.conv2 = tf.layers.Conv2D(filters=channel_2, kernel_size=(3, 3), padding='same', strides=[1, 1],
                                      kernel_initializer=initializer, data_format='channels_last',
                                      activation=tf.nn.relu, name='conv2')
        self.outputs = tf.layers.Dense(num_classes, kernel_initializer=initializer, name='outputs')
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def call(self, x, training=None):
        scores = None
        ########################################################################
        # Implement the forward pass for a three-layer ConvNet. You      #
        # should use the layer objects defined in the __init__ method.         #
        ########################################################################
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv2_out_flatten = tf.layers.flatten(conv2_out)
        scores = self.outputs(conv2_out_flatten)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return scores


def test_three_layer_conv_net():
    tf.reset_default_graph()

    channel_1, channel_2, num_classes = 12, 8, 10
    model = ThreeLayerConvNet(channel_1, channel_2, num_classes)
    with tf.device(device):
        x = tf.zeros((64, 32, 32, 3))
        scores = model(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape)


def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.

    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for

    Returns: Nothing, but prints progress during trainingn
    """
    tf.reset_default_graph()
    with tf.device(device):
        # Construct the computational graph we will use to train the model. We
        # use the model_init_fn to construct the model, declare placeholders for
        # the data and labels
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])

        # We need a place holder to explicitly specify if the model is in the training
        # phase or not. This is because a number of layers behaves differently in
        # training and in testing, e.g., dropout and batch normalization.
        is_training = tf.placeholder(tf.bool, name='is_training')

        # Use the model function to build the forward pass.
        scores = model_init_fn(x, is_training)

        # Compute the loss like we did in part II
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(loss)
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss

        # Use the optimizer_fn to construct an Optimizer, then use the optimizer
        # to set up the training step. Asking TensorFlow to evaluate the
        # train_op returned by optimizer.minimize(loss) will cause us to make a
        # single update step using the current minibatch of data.

        # Note that we use tf.control_dependencies to force the model to run
        # the tf.GraphKeys.UPDATE_OPS at each training step. tf.GraphKeys.UPDATE_OPS
        # holds the operators that update the states of the network.
        # For example, the tf.layers.batch_normalization function adds the running mean
        # and variance update operators to tf.GraphKeys.UPDATE_OPS.
        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    # Now we can run the computational graph many times to train the model.
    # When we call sess.run we ask it to evaluate train_op, which causes the
    # model to update.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d:' % epoch)
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, is_training: 1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))
                    check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                    print()
                t += 1


def fc_model_init_fn(inputs, is_training):
    hidden_size, num_classes = 4000, 10
    # return TwoLayerFC(hidden_size, num_classes)(inputs)
    return two_layer_fc_functional(inputs, hidden_size, num_classes)


def fc_optimizer_init_fn():
    learning_rate = 1e-2
    # return tf.train.GradientDescentOptimizer(learning_rate)
    return tf.train.GradientDescentOptimizer(learning_rate)


def conv_model_init_fn(inputs, is_training):
    model = None
    channel_1, channel_2, num_classes = 32, 16, 10
    ############################################################################
    # Complete the implementation of model_fn.                           #
    ############################################################################
    model = ThreeLayerConvNet(channel_1, channel_2, num_classes)
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return model(inputs)


def conv_optimizer_init_fn():
    optimizer = None
    learning_rate = 3e-3
    momentum = 0.9
    ############################################################################
    # Complete the implementation of model_fn.                           #
    ############################################################################
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return optimizer


"""Part IV: Keras Sequential API"""


def sequential_fc_model_init_fn(inputs, is_training):
    input_shape = (32, 32, 3)
    hidden_layer_size, num_classes = 4000, 10
    initializer = tf.variance_scaling_initializer(scale=2.0)
    layers = [
        tf.layers.Flatten(input_shape=input_shape),
        tf.layers.Dense(hidden_layer_size, activation=tf.nn.relu, kernel_initializer=initializer),
        tf.layers.Dense(num_classes, kernel_initializer=initializer),
    ]
    model = tf.keras.Sequential(layers)
    return model(inputs)


def sequential_fc_optimizer_init_fn():
    learning_rate = 1e-2
    return tf.train.GradientDescentOptimizer(learning_rate)


def sequential_conv_model_init_fn(inputs, is_training):
    model = None
    ############################################################################
    # Construct a three-layer ConvNet using tf.keras.Sequential.         #
    ############################################################################
    initializer = tf.variance_scaling_initializer(scale=2.0)
    layers = [
        tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same', strides=[1, 1],
                         kernel_initializer=initializer, data_format='channels_last',
                         activation=tf.nn.relu, name='conv1'),
        tf.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=[1, 1],
                         kernel_initializer=initializer, data_format='channels_last',
                         activation=tf.nn.relu, name='conv2'),
        tf.layers.Flatten(),
        tf.layers.Dense(10, kernel_initializer=initializer, name='outputs')

    ]
    model = tf.keras.Sequential(layers)
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################
    return model(inputs)


def sequential_conv_optimizer_init_fn():
    optimizer = None
    learning_rate = 5e-4
    ############################################################################
    # Complete the implementation of model_fn.                           #
    ############################################################################
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return optimizer


"""Part V: CIFAR-10 open-ended challenge"""


def my_model_init_fn(inputs, is_training):
    initializer = tf.variance_scaling_initializer(scale=2.0)
    reg = 1e-5
    layers = [
        tf.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=1, kernel_initializer=initializer,
                         kernel_regularizer=tf.keras.regularizers.l2(reg), name='conv_1_1'),
        tf.layers.BatchNormalization(name='bn_1_1'),
        tf.keras.layers.Activation(activation='relu', name='relu_1_1'),
        tf.layers.Dropout(rate=0.5, name='drop_1_1'),
        tf.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=1, kernel_initializer=initializer,
                         kernel_regularizer=tf.keras.regularizers.l2(reg), name='conv_1_2'),
        tf.layers.BatchNormalization(name='bn_1_2'),
        tf.keras.layers.Activation(activation='relu', name='relu_1_2'),
        tf.layers.Dropout(rate=0.5, name='drop_1_2'),
        tf.layers.MaxPooling2D(pool_size=2, strides=2, name='pool_1'),

        tf.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides=1, kernel_initializer=initializer,
                         kernel_regularizer=tf.keras.regularizers.l2(reg), name='conv_2_1'),
        tf.layers.BatchNormalization(name='bn_2_1'),
        tf.keras.layers.Activation(activation='relu', name='relu_2_1'),
        tf.layers.Dropout(rate=0.5, name='drop_2_1'),
        tf.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides=1, kernel_initializer=initializer,
                         kernel_regularizer=tf.keras.regularizers.l2(reg), name='conv_2_2'),
        tf.layers.BatchNormalization(name='bn_2_2'),
        tf.keras.layers.Activation(activation='relu', name='relu_2_2'),
        tf.layers.Dropout(rate=0.5, name='drop_2_2'),
        tf.layers.MaxPooling2D(pool_size=2, strides=2, name='pool_2'),

        tf.layers.Conv2D(filters=256, kernel_size=3, padding='same', strides=1, kernel_initializer=initializer,
                         kernel_regularizer=tf.keras.regularizers.l2(reg), name='conv_3_1'),
        tf.layers.BatchNormalization(name='bn_3_1'),
        tf.keras.layers.Activation(activation='relu', name='relu_3_1'),
        tf.layers.Dropout(rate=0.5, name='drop_3_1'),
        tf.layers.Conv2D(filters=256, kernel_size=3, padding='same', strides=1, kernel_initializer=initializer,
                         kernel_regularizer=tf.keras.regularizers.l2(reg), name='conv_3_2'),
        tf.layers.BatchNormalization(name='bn_3_2'),
        tf.keras.layers.Activation(activation='relu', name='relu_3_2'),
        tf.layers.Dropout(rate=0.5, name='drop_3_2'),
        tf.layers.MaxPooling2D(pool_size=2, strides=2, name='pool_3'),

        tf.layers.Flatten(),
        tf.layers.Dense(10, kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(reg),
                        name='outputs'),
    ]
    model = tf.keras.Sequential(layers)
    return model(inputs)


def my_optimizer_init_fn():
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    return optimizer


if __name__ == '__main__':
    """Part I: Preparation"""
    # Invoke the above function to get our data.
    # NHW = (0, 1, 2)
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    # print('Train data shape: ', X_train.shape)
    # print('Train labels shape: ', y_train.shape, y_train.dtype)
    # print('Validation data shape: ', X_val.shape)
    # print('Validation labels shape: ', y_val.shape)
    # print('Test data shape: ', X_test.shape)
    # print('Test labels shape: ', y_test.shape)
    train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
    val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
    test_dset = Dataset(X_test, y_test, batch_size=64)
    # We can iterate through a dataset like this:
    # for t, (x, y) in enumerate(train_dset):
    #     print(t, x.shape, y.shape)
    #     if t > 5:
    #         break

    # Set up some global variables
    USE_GPU = True
    if USE_GPU:
        device = '/device:GPU:0'
    else:
        device = '/cpu:0'
    # Constant to control how often we print when training models
    print_every = 200
    # print('Using device: ', device)
    """Part II: Barebone TensorFlow"""
    # Train a Two-Layer Network
    # learning_rate = 1e-2
    # train_part2(two_layer_fc, two_layer_fc_init, learning_rate)
    # Train a three-layer ConvNet
    # learning_rate = 3e-3
    # train_part2(three_layer_convnet, three_layer_convnet_init, learning_rate)
    """Part III: Keras API"""
    # test_two_layer_fc()
    # test_two_layer_fc_functional()
    # test_three_layer_conv_net()
    # train_part34(fc_model_init_fn, fc_optimizer_init_fn)
    # train_part34(conv_model_init_fn, conv_optimizer_init_fn)
    """Part IV: Keras Sequential API"""
    # train_part34(sequential_fc_model_init_fn, sequential_fc_optimizer_init_fn)
    # train_part34(sequential_conv_model_init_fn, sequential_conv_optimizer_init_fn)
    """Part V: CIFAR-10 open-ended challenge"""
    train_part34(my_model_init_fn, my_optimizer_init_fn, num_epochs=10)
