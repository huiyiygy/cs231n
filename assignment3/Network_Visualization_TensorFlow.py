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
from scipy.ndimage.filters import gaussian_filter1d

from cs231n.classifiers.squeezenet import SqueezeNet
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
    saliency = sess.run(saliency_grad, {model.image: X, model.labels: y})[0]
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


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, a numpy array of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """

    # Make a copy of the input that we will modify
    X_fooling = X.copy()

    # Step size for the update
    learning_rate = 1

    ##############################################################################
    # Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient *ascent* on the target class score, using #
    # the model.scores Tensor to get the class scores for the model.image.   #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop, where in each iteration, you make an     #
    # update to the input image X_fooling (don't modify X). The loop should      #
    # stop when the predicted class for the input is the same as target_y.       #
    #                                                                            #
    # HINT: It's good practice to define your TensorFlow graph operations        #
    # outside the loop, and then just make sess.run() calls in each iteration.   #
    #                                                                            #
    # HINT 2: For most examples, you should be able to generate a fooling image  #
    # in fewer than 100 iterations of gradient ascent. You can print your        #
    # progress over iterations to check your algorithm.                          #
    ##############################################################################
    for i in range(100):
        # score = model.scores[0, target_y]
        score = tf.gather_nd(model.scores, tf.stack((tf.range(X.shape[0]), [target_y]), axis=1))
        saliency_grads = tf.gradients(score, model.image)
        scores, saliency_grads = sess.run([model.scores, saliency_grads], {model.image: X_fooling, model.labels: [target_y]})
        score = np.argmax(scores[0])
        if score == target_y:
            break
        print("Iteration %d / 100: not fooled, score=%d" % (i + 1, score))
        saliency_grad = saliency_grads[0]
        dX = learning_rate * saliency_grad / np.linalg.norm(saliency_grad)
        X_fooling += dX
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling


def generate_fooling_images():
    idx = 4
    Xi = X[idx][None]
    target_y = 89
    X_fooling = make_fooling_image(Xi, target_y, model)

    # Make sure that X_fooling is classified as y_target
    scores = sess.run(model.scores, {model.image: X_fooling})
    assert scores[0].argmax() == target_y, 'The network is not fooled!'

    # Show original image, fooling image, and difference
    orig_img = deprocess_image(Xi[0])
    fool_img = deprocess_image(X_fooling[0])
    # Rescale
    plt.subplot(2, 2, 1)
    plt.imshow(orig_img)
    plt.axis('off')
    plt.title(class_names[y[idx]])
    plt.subplot(2, 2, 2)
    plt.imshow(fool_img)
    plt.title(class_names[target_y])
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title('Difference')
    plt.imshow(deprocess_image((Xi - X_fooling)[0]))
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title('Magnified difference (10x)')
    plt.imshow(deprocess_image(10 * (Xi - X_fooling)[0]))
    plt.axis('off')


def blur_image(X, sigma=1.0):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X


def create_class_visualization(target_y, model, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    # We use a single image of random noise as a starting point
    X = 255 * np.random.rand(224, 224, 3)
    X = preprocess_image(X)[None]

    ########################################################################
    # Compute the loss and the gradient of the loss with respect to  #
    # the input image, model.image. We compute these outside the loop so   #
    # that we don't have to recompute the gradient graph at each iteration #
    #                                                                      #
    # Note: loss and grad should be TensorFlow Tensors, not numpy arrays!  #
    #                                                                      #
    # The loss is the score for the target label, target_y. You should     #
    # use model.scores to get the scores, and tf.gradients to compute  #
    # gradients. Don't forget the (subtracted) L2 regularization term!     #
    ########################################################################
    loss = None  # scalar loss
    grad = None  # gradient of loss with respect to model.image, same size as model.image
    score = tf.gather_nd(model.scores, tf.stack((tf.range(X.shape[0]), [target_y]), axis=1))
    loss = score - l2_reg * tf.reduce_sum(tf.norm(model.image, ord=2) ** 2)
    grad = tf.gradients(loss, model.image)[0]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = np.random.randint(-max_jitter, max_jitter + 1, 2)
        X = np.roll(np.roll(X, ox, 1), oy, 2)

        ########################################################################
        # Use sess to compute the value of the gradient of the score for #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. You should use   #
        # the grad variable you defined above.                                 #
        #                                                                      #
        # Be very careful about the signs of elements in your code.            #
        ########################################################################
        gradient = sess.run([grad], {model.image: X, model.labels: [target_y]})
        gradient = gradient[0]
        dx = learning_rate * gradient / np.linalg.norm(gradient)
        X += dx
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, 1), -oy, 2)

        # As a regularizer, clip and periodically blur
        X = np.clip(X, -SQUEEZENET_MEAN / SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD)
        if t % blur_every == 0:
            X = blur_image(X, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess_image(X[0]))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()
    return X


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

    # mask = np.arange(5)
    # show_saliency_maps(X, y, mask)

    # generate_fooling_images()

    # target_y = 76  # Tarantula
    # target_y = 78 # Tick
    # target_y = 187 # Yorkshire Terrier
    # target_y = 683 # Oboe
    # target_y = 366 # Gorilla
    # target_y = 604 # Hourglass
    # out = create_class_visualization(target_y, model)

    target_y = np.random.randint(1000)
    print(class_names[target_y])
    params = {'l2_reg': 1e-4, 'learning_rate': 100, 'num_iterations': 500, 'show_every': 50}
    X = create_class_visualization(target_y, model, **params)
