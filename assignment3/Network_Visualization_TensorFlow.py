# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file: Network_Visualization_TensorFlow.py
@time: 2018/12/10 19:52
"""
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.data_utils import load_tiny_imagenet
from cs231n.data_utils import load_imagenet_val
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.
    correct_scores = tf.gather_nd(model.scores,
                                  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
    ###############################################################################
    # Produce the saliency maps over a batch of images.                     #
    #                                                                             #
    # 1) Compute the “loss” using the correct scores tensor provided for you.     #
    #    (We'll combine losses across a batch by summing)                         #
    # 2) Use tf.gradients to compute the gradient of the loss with respect        #
    #    to the image (accessible via model.image).                               #
    # 3) Compute the actual value of the gradient by a call to sess.run().        #
    #    You will need to feed in values for the placeholders model.image and     #
    #    model.labels.                                                            #
    # 4) Finally, process the returned gradient to compute the saliency map.      #
    ###############################################################################
    saliency_grad = tf.gradients(correct_scores, model.image)
    saliency = sess.run(saliency_grad, {model.image:X, model.labels:y})[0]
    saliency = np.abs(saliency)
    saliency = np.max(saliency, axis=-1)  # get the maximum value in all three channels
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def show_saliency_maps(X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()


if __name__ == "__main__":
    tf.reset_default_graph()
    sess = get_session()

    SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'
    if not os.path.exists(SAVE_PATH + ".index"):
        raise ValueError("You need to download SqueezeNet!")
    model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

    X_raw, y, class_names = load_imagenet_val(num=5)

    # plt.figure(figsize=(12, 6))
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(X_raw[i])
    #     plt.title(class_names[y[i]])
    #     plt.axis('off')
    # plt.gcf().tight_layout()

    X = np.array([preprocess_image(img) for img in X_raw])

    mask = np.arange(5)
    show_saliency_maps(X, y, mask)
