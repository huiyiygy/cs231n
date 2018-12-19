# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:GANs_TensorFlow.py
@time:2018/12/19 21:02
"""
import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


class MNIST(object):
    def __init__(self, batch_size, shuffle=False):
        """
        Construct an iterator object over the MNIST data

        Inputs:
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        train, _ = tf.keras.datasets.mnist.load_data()
        X, y = train
        X = X.astype(np.float32) / 255
        X = X.reshape((X.shape[0], -1))
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))


def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.maximum(x, alpha*x)


def test_leaky_relu(x, y_true):
    tf.reset_default_graph()
    with get_session() as sess:
        y_tf = leaky_relu(tf.constant(x))
        y = sess.run(y_tf)
        print('Maximum error: %g' % rel_error(y_true, y))


def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.

    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate

    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    return tf.random_uniform([batch_size, dim], -1, 1)


def test_sample_noise():
    batch_size = 3
    dim = 4
    tf.reset_default_graph()
    with get_session() as sess:
        z = sample_noise(batch_size, dim)
        # Check z has the correct shape
        assert z.get_shape().as_list() == [batch_size, dim]
        # Make sure z is a Tensor and not a numpy array
        assert isinstance(z, tf.Tensor)
        # Check that we get different noise for different evaluations
        z1 = sess.run(z)
        z2 = sess.run(z)
        assert not np.array_equal(z1, z2)
        # Check that we get the correct range
        assert np.all(z1 >= -1.0) and np.all(z1 <= 1.0)
        print("All tests passed!")


# def discriminator(x):
#     """Compute discriminator score for a batch of input images.
#
#     Inputs:
#     - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
#
#     Returns:
#     TensorFlow Tensor with shape [batch_size, 1], containing the score
#     for an image being real for each input image.
#     """
#     with tf.variable_scope("discriminator"):
#         layer1 = tf.layers.dense(x, 256)
#         layer1 = leaky_relu(layer1, alpha=0.01)
#         layer2 = tf.layers.dense(layer1, 256)
#         layer2 = leaky_relu(layer2, alpha=0.01)
#         logits = tf.layers.dense(layer2, 1)
#         return logits


def test_discriminator(true_count=267009):
    tf.reset_default_graph()
    with get_session() as sess:
        y = discriminator(tf.ones((2, 784)))
        cur_count = count_params()
        if cur_count != true_count:
            print('Incorrect number of parameters in discriminator. {0} instead of {1}. Check your achitecture.'.format(cur_count,true_count))
        else:
            print('Correct number of parameters in discriminator.')


# def generator(z):
#     """Generate images from a random noise vector.
#
#     Inputs:
#     - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
#
#     Returns:
#     TensorFlow Tensor of generated images, with shape [batch_size, 784].
#     """
#     with tf.variable_scope("generator"):
#         layer1 = tf.layers.dense(z, 1024, activation=tf.nn.relu)
#         layer2 = tf.layers.dense(layer1, 1024, activation=tf.nn.relu)
#         img = tf.layers.dense(layer2, 784, activation=tf.nn.tanh)
#         return img


def test_generator(true_count=1858320):
    tf.reset_default_graph()
    with get_session() as sess:
        y = generator(tf.ones((1, 4)))
        cur_count = count_params()
        if cur_count != true_count:
            print('Incorrect number of parameters in generator. {0} instead of {1}. Check your achitecture.'.format(cur_count,true_count))
        else:
            print('Correct number of parameters in generator.')


def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar

    HINT: for the discriminator loss, you'll want to do the averaging separately for
    its two components, and then add them together (instead of averaging once at the very end).
    """
    D_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real)
    D_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake)
    D_loss = tf.reduce_mean(D_real_loss) + tf.reduce_mean(D_fake_loss)

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
    return D_loss, G_loss


def test_gan_loss(logits_real, logits_fake, d_loss_true, g_loss_true):
    tf.reset_default_graph()
    with get_session() as sess:
        d_loss, g_loss = sess.run(gan_loss(tf.constant(logits_real), tf.constant(logits_fake)))
    print("Maximum error in d_loss: %g" % rel_error(d_loss_true, d_loss))
    print("Maximum error in g_loss: %g" % rel_error(g_loss_true, g_loss))


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.

    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)

    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    D_solver = tf.train.AdamOptimizer(learning_rate, beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate, beta1)
    return D_solver, G_solver


def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, show_every=2, print_every=1, batch_size=128, num_epoch=10):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """
    # compute the number of iterations we need
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    for epoch in range(num_epoch):
        # every show often, show a sample result
        if epoch % show_every == 0:
            samples = sess.run(G_sample)
            fig = show_images(samples[:16])
            plt.show()
            print()
        for (minibatch, minbatch_y) in mnist:
            # run a batch of data through the network
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if epoch % print_every == 0:
            print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
    print('Final images')
    samples = sess.run(G_sample)

    fig = show_images(samples[:16])
    plt.show()


def lsgan_loss(scores_real, scores_fake):
    """Compute the Least Squares GAN loss.

    Inputs:
    - scores_real: Tensor, shape [batch_size, 1], output of discriminator
        The score for each real image
    - scores_fake: Tensor, shape[batch_size, 1], output of discriminator
        The score for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    D_loss = 0.5 * tf.reduce_mean(tf.square(scores_real-1)) + 0.5 * tf.reduce_mean(tf.square(scores_fake))
    G_loss = 0.5 * tf.reduce_mean(tf.square(scores_fake - 1))
    return D_loss, G_loss


def test_lsgan_loss(score_real, score_fake, d_loss_true, g_loss_true):
    with get_session() as sess:
        d_loss, g_loss = sess.run(
            lsgan_loss(tf.constant(score_real), tf.constant(score_fake)))
    print("Maximum error in d_loss: %g"%rel_error(d_loss_true, d_loss))
    print("Maximum error in g_loss: %g"%rel_error(g_loss_true, g_loss))


def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        x = tf.reshape(x, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, strides=(1, 1), padding='valid', activation=leaky_relu)
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(pool1, 64, 5, strides=(1, 1), padding='valid', activation=leaky_relu)
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
        flatten1 = tf.layers.flatten(pool2)
        fc1 = tf.layers.dense(flatten1, 4 * 4 * 64, activation=leaky_relu)
        logits = tf.layers.dense(fc1, 1)
        return logits


def generator(z):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):
        fc1 = tf.layers.dense(z, 1024, activation=tf.nn.relu)
        bn1 = tf.layers.batch_normalization(fc1, training=True)
        fc2 = tf.layers.dense(bn1, 7 * 7 * 128, activation=tf.nn.relu)
        bn2 = tf.layers.batch_normalization(fc2, training=True)
        img = tf.reshape(bn2, [-1, 7, 7, 128])
        conv1 = tf.layers.conv2d_transpose(img, 64, 4, 2, padding='same', activation=tf.nn.relu)
        bn3 = tf.layers.batch_normalization(conv1, training=True)
        conv2 = tf.layers.conv2d_transpose(bn3, 1, 4, 2, padding='same', activation=tf.tanh)
        img = tf.reshape(conv2, [-1, 784])
        return img


if __name__ == "__main__":
    answers = np.load('gan-checks-tf.npz')

    # test_leaky_relu(answers['lrelu_x'], answers['lrelu_y'])
    # test_sample_noise()

    # test_discriminator()
    # test_generator()

    # test_gan_loss(answers['logits_real'], answers['logits_fake'],
    #               answers['d_loss_true'], answers['g_loss_true'])

    """Training a GAN"""
    # tf.reset_default_graph()
    # # number of images for each batch
    # batch_size = 128
    # # our noise dimension
    # noise_dim = 96
    #
    # # placeholder for images from the training dataset
    # x = tf.placeholder(tf.float32, [None, 784])
    # # random noise fed into our generator
    # z = sample_noise(batch_size, noise_dim)
    # # generated images
    # G_sample = generator(z)
    #
    # with tf.variable_scope("") as scope:
    #     # scale images to be -1 to 1
    #     logits_real = discriminator(preprocess_img(x))
    #     # Re-use discriminator weights on new inputs
    #     scope.reuse_variables()
    #     logits_fake = discriminator(G_sample)
    #
    # # Get the list of variables for the discriminator and generator
    # D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    # G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    #
    # # get our solver
    # D_solver, G_solver = get_solvers()

    # # get our loss
    # D_loss, G_loss = gan_loss(logits_real, logits_fake)
    #
    # # setup training steps
    # D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    # G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
    # D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
    # G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
    #
    # with get_session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step)

    """Least Squares GAN"""
    # test_lsgan_loss(answers['logits_real'], answers['logits_fake'],
    #                 answers['d_loss_lsgan_true'], answers['g_loss_lsgan_true'])

    # D_loss, G_loss = lsgan_loss(logits_real, logits_fake)
    # D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    # G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
    #
    # with get_session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step)

    """Deep Convolutional GANs"""
    tf.reset_default_graph()

    batch_size = 128
    # our noise dimension
    noise_dim = 96

    # placeholders for images from the training dataset
    x = tf.placeholder(tf.float32, [None, 784])
    z = sample_noise(batch_size, noise_dim)
    # generated images
    G_sample = generator(z)

    with tf.variable_scope("") as scope:
        # scale images to be -1 to 1
        logits_real = discriminator(preprocess_img(x))
        # Re-use discriminator weights on new inputs
        scope.reuse_variables()
        logits_fake = discriminator(G_sample)

    # Get the list of variables for the discriminator and generator
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

    D_solver, G_solver = get_solvers()
    D_loss, G_loss = gan_loss(logits_real, logits_fake)
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
    D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
    G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

    with get_session() as sess:
        sess.run(tf.global_variables_initializer())
        run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, num_epoch=5)
