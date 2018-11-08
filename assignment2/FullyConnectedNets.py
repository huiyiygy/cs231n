# -*- coding:utf-8 -*-
"""
@author:HuiYi or 会意
@file:FullyConnectedNets.py
@time:2018/09/01 16:30
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import cs231n.layers as layers
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
from cs231n.optim import *


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def check_affine_forward():
    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)  # 返回给定轴上的数组元素的乘积
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)  # 在指定的间隔内返回均匀间隔的数字
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    out, _ = layers.affine_forward(x, w, b)
    correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                            [3.25553199, 3.5141327, 3.77273342]])

    # Compare your output with ours. The error should be around e-9 or less.
    print('Testing affine_forward function:')
    print('difference: ', rel_error(out, correct_out))


def check_affine_backward():
    np.random.seed(231)
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    dx_num = eval_numerical_gradient_array(lambda x: layers.affine_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: layers.affine_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: layers.affine_forward(x, w, b)[0], b, dout)

    _, cache = layers.affine_forward(x, w, b)
    dx, dw, db = layers.affine_backward(dout, cache)

    # The error should be around e-10 or less
    print('Testing affine_backward function:')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))


def check_relu_forward():
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    out, _ = layers.relu_forward(x)
    correct_out = np.array([[0., 0., 0., 0., ],
                            [0., 0., 0.04545455, 0.13636364, ],
                            [0.22727273, 0.31818182, 0.40909091, 0.5, ]])
    # Compare your output with ours. The error should be on the order of e-8
    print('Testing relu_forward function:')
    print('difference: ', rel_error(out, correct_out))


def check_relu_backward():
    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: layers.relu_forward(x)[0], x, dout)

    _, cache = layers.relu_forward(x)
    dx = layers.relu_backward(dout, cache)

    # The error should be on the order of e-12
    print('Testing relu_backward function:')
    print('dx error: ', rel_error(dx_num, dx))


def check_affine_relu_forward_and_backward_gradient():
    np.random.seed(231)
    x = np.random.randn(2, 3, 4)
    w = np.random.randn(12, 10)
    b = np.random.randn(10)
    dout = np.random.randn(2, 10)

    out, cache = affine_relu_forward(x, w, b)
    dx, dw, db = affine_relu_backward(dout, cache)

    dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

    # Relative error should be around e-10 or less
    print('Testing affine_relu_forward and affine_relu_backward:')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))


def check_svm_and_softmax_loss():
    np.random.seed(231)
    num_classes, num_input = 10, 50
    x = 0.001 * np.random.rand(num_input, num_classes)
    y = np.random.randint(num_classes, size=num_input)

    svm_dx_num = eval_numerical_gradient(lambda x: layers.svm_loss(x, y)[0], x, verbose=False)
    svm_loss, svm_dx = layers.svm_loss(x, y)

    # Test svm_loss function. Loss should be around 9 and dx error should be around the order of e-9
    print('Testing svm_loss:')
    print('loss: ', svm_loss)
    print('dx error: ', rel_error(svm_dx_num, svm_dx))

    softmax_dx_num = eval_numerical_gradient(lambda x: layers.softmax_loss(x, y)[0], x, verbose=False)
    softmax_loss, softmax_dx = layers.softmax_loss(x, y)

    # Test softmax_loss function. Loss should be close to 2.3 and dx error should be around e-8
    print('\nTesting softmax_loss:')
    print('loss: ', softmax_loss)
    print('dx error: ', rel_error(softmax_dx_num, softmax_dx))


def check_two_layer_network():
    np.random.seed(231)
    N, D, H, C = 3, 5, 50, 7
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=N)

    std = 1e-3
    model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

    print('Testing initialization...')
    W1_std = abs(model.params['W1'].std() - std)
    b1 = model.params['b1']
    W2_std = abs(model.params['W2'].std() - std)
    b2 = model.params['b2']
    assert W1_std < std / 10, 'First layer weight do not seem right'
    assert np.all(b1 == 0), 'First layer biases do not seem right'
    assert W2_std < std / 10, 'Second layer weight do not seem right'
    assert np.all(b2 == 0), 'First layer biases do not seem right'

    print('Testing test-time forward pass...')
    model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
    model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
    model.params['W2'] = np.linspace(-0.3, 0.4, num=H * C).reshape(H, C)
    model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
    X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T  # 注意此处的做法
    scores = model.loss(X)
    correct_scores = np.asarray(
        [[11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
         [12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128, 15.49994135, 16.18839143],
         [12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822, 15.66781506, 16.2846319]])
    scores_diff = np.abs(scores - correct_scores).sum()
    assert scores_diff < 1e-6, 'Problem with test-time forward pass'

    print('Testing training loss(no regularization)...')
    y = np.asarray([0, 5, 1])
    loss, grad = model.loss(X, y)
    correct_loss = 3.4702243556
    assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

    print('Testing training loss with regularization...')
    model.reg = 1.0
    loss, grad = model.loss(X, y)
    correct_loss = 26.5948426952
    assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

    # Errors should be around e-7 or less
    for reg in [0.0, 0.7]:
        print('Running numeric gradient check with reg=', reg)
        model.reg = reg
        loss, grads = model.loss(X, y)
        for name in sorted(grads):
            f = lambda _: model.loss(X,y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))


def check_solver():
    data = get_CIFAR10_data()
    model = TwoLayerNet()
    solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-3},
                    lr_decay=0.95, num_epochs=20, batch_size=256, print_every=100)
    solver.train()

    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()


def check_loss_and_gradient():
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = FullyConnectedNet([H1, H2], input_dims=D, num_classes=C,
                                  reg=reg, weight_scale=5e-2, dtype=np.float64)

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        # Most of the errors should be on the order of e-7 or smaller.
        # NOTE: It is fine however to see an error for W2 on the order of e-5
        # for the check when reg = 0.0
        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))


def three_layer_net_overfit_50_training_examples():
    """Use a three-layer Net to overfit 50 training examples by tweaking just the learning rate and initialization scale."""
    data = get_CIFAR10_data()
    num_train = 50
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    weight_scale = 1e-2
    learning_rate = 1e-2
    model = FullyConnectedNet([100, 100],
                              weight_scale=weight_scale, dtype=np.float64)
    solver = Solver(model, small_data,
                    print_every=10, num_epochs=20, batch_size=25,
                    update_rule='sgd',
                    optim_config={'learning_rate': learning_rate}
                    )
    solver.train()
    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, '-o')
    plt.title('Training loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label='train acc')
    plt.plot(solver.val_acc_history, '-o', label='val acc')
    plt.title('Accuracy history')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left', ncol=2)
    plt.gcf().set_size_inches(15, 10)
    plt.show()


def five_layer_net_overfit_50_training_examples():
    # Use a five-layer Net to overfit 50 training examples by tweaking just the learning rate and initialization scale.
    data = get_CIFAR10_data()
    num_train = 50
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    learning_rate = 2e-3
    weight_scale = 1e-5
    model = FullyConnectedNet([100, 100, 100, 100],
                              weight_scale=weight_scale, dtype=np.float64)
    solver = Solver(model, small_data,
                    print_every=10, num_epochs=20, batch_size=25,
                    update_rule='sgd',
                    optim_config={
                        'learning_rate': learning_rate,
                    }
                    )
    solver.train()
    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, '-o')
    plt.title('Training loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label='train acc')
    plt.plot(solver.val_acc_history, '-o', label='val acc')
    plt.title('Accuracy history')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left', ncol=2)
    plt.gcf().set_size_inches(15, 10)
    plt.show()


def check_sgd_momentum():
    """You should see errors less than e-8"""
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    v = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-3, 'velocity': v}
    next_w, _ = sgd_momentum(w, dw, config=config)

    expected_next_w = np.asarray([
        [0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789],
        [0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526],
        [0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263],
        [1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096]])
    expected_velocity = np.asarray([
        [0.5406, 0.55475789, 0.56891579, 0.58307368, 0.59723158],
        [0.61138947, 0.62554737, 0.63970526, 0.65386316, 0.66802105],
        [0.68217895, 0.69633684, 0.71049474, 0.72465263, 0.73881053],
        [0.75296842, 0.76712632, 0.78128421, 0.79544211, 0.8096]])

    # Should see relative errors around e-8 or less
    print('next_w error: ', rel_error(next_w, expected_next_w))
    print('velocity error: ', rel_error(expected_velocity, config['velocity']))


def check_rmsprop():
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    cache = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-2, 'cache': cache}
    next_w, _ = rmsprop(w, dw, config=config)

    expected_next_w = np.asarray([
        [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
        [-0.132737, -0.08078555, -0.02881884, 0.02316247, 0.07515774],
        [0.12716641, 0.17918792, 0.23122175, 0.28326742, 0.33532447],
        [0.38739248, 0.43947102, 0.49155973, 0.54365823, 0.59576619]])
    expected_cache = np.asarray([
        [0.5976, 0.6126277, 0.6277108, 0.64284931, 0.65804321],
        [0.67329252, 0.68859723, 0.70395734, 0.71937285, 0.73484377],
        [0.75037008, 0.7659518, 0.78158892, 0.79728144, 0.81302936],
        [0.82883269, 0.84469141, 0.86060554, 0.87657507, 0.8926]])

    # You should see relative errors around e-7 or less
    print('next_w error: ', rel_error(expected_next_w, next_w))
    print('cache error: ', rel_error(expected_cache, config['cache']))


def check_adam():
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    m = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)
    v = np.linspace(0.7, 0.5, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
    next_w, _ = adam(w, dw, config=config)

    expected_next_w = np.asarray([
        [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
        [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929],
        [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969],
        [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459]])
    expected_v = np.asarray([
        [0.69966, 0.68908382, 0.67851319, 0.66794809, 0.65738853, ],
        [0.64683452, 0.63628604, 0.6257431, 0.61520571, 0.60467385, ],
        [0.59414753, 0.58362676, 0.57311152, 0.56260183, 0.55209767, ],
        [0.54159906, 0.53110598, 0.52061845, 0.51013645, 0.49966, ]])
    expected_m = np.asarray([
        [0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474],
        [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316],
        [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158],
        [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85]])

    # You should see relative errors around e-7 or less
    print('next_w error: ', rel_error(expected_next_w, next_w))
    print('v error: ', rel_error(expected_v, config['v']))
    print('m error: ', rel_error(expected_m, config['m']))


def compare_different_optim():
    data = get_CIFAR10_data()
    num_train = 5000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    solvers = {}

    for update_rule in ['sgd', 'sgd_momentum', 'adam', 'rmsprop']:
        print('running with ', update_rule)
        model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=3e-3)

        solver = Solver(model, small_data,
                        num_epochs=10, batch_size=100,
                        update_rule=update_rule,
                        optim_config={'learning_rate': 3e-4},
                        verbose=True)
        solvers[update_rule] = solver
        solver.train()
        print()

    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    for update_rule, solver in list(solvers.items()):
        plt.subplot(3, 1, 1)
        plt.plot(solver.loss_history, 'o', label=update_rule)
        plt.legend(loc='upper right', ncol=4)

        plt.subplot(3, 1, 2)
        plt.plot(solver.train_acc_history, '-o', label=update_rule)
        plt.legend(loc='upper left', ncol=4)

        plt.subplot(3, 1, 3)
        plt.plot(solver.val_acc_history, '-o', label=update_rule)
        plt.legend(loc='upper left', ncol=4)

    plt.gcf().set_size_inches(15, 15)
    plt.show()


def find_best_model():
    data = get_CIFAR10_data()
    best_solver = None
    best_solver_weight_scale = 0.0
    best_val_acc = 0.0

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    weight_scales = [1e-4, 1e-3, 1e-2, 1e-1]

    for learning_rate in learning_rates:
        for weight_scale in weight_scales:
            print('running with learning_rate: %f, weight_scale: %f' % (learning_rate, weight_scale))
            t1 = time.time()
            model = FullyConnectedNet([600, 500, 400, 300, 200, 100], reg=0.01, weight_scale=weight_scale, dropout=0.5, normalization='batchnorm')

            solver = Solver(model, data,
                            num_epochs=10, batch_size=200,
                            update_rule='adam',
                            optim_config={'learning_rate': learning_rate},
                            verbose=False)
            solver.train()
            if solver.best_val_acc > best_val_acc:
                best_val_acc = solver.best_val_acc
                best_solver = solver
                best_solver_weight_scale = weight_scale
            t2 = time.time()
            print('time used: %.2f s' % (t2-t1))

    print("The best Solver's parameters are: learning_rate: %f, weight_scale: %f" %
          (best_solver.optim_config['learning_rate'], best_solver_weight_scale))
    print('The best validation accuracy is:', best_solver.best_val_acc)

    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')
    plt.plot(best_solver.loss_history)

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(best_solver.train_acc_history, label='Training accuracy')
    plt.plot(best_solver.val_acc_history, label='Validation accuracy')
    plt.legend(loc='upper left', ncol=2)
    plt.gcf().set_size_inches(15, 10)
    plt.show()

    y_test_pred = np.argmax(best_solver.model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(best_solver.model.loss(data['X_val']), axis=1)
    print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
    print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())


if __name__ == '__main__':
    find_best_model()
