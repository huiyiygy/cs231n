# -*- coding:utf-8 -*-
"""
@author:HuiYi or 会意
@file:solver.py
@time:2018/09/01 16:50
"""
import pickle as pickle
import numpy as np
from cs231n import optim


class Solver:
    """
    Solver封装了训练分类模型所需的所有逻辑。 Solver使用optim.py中定义的不同更新规则执行随机梯度下降。

    Solver接受训练和验证数据以及标签，因此它可以定期检查训练和验证数据的分类准确性，以注意过度拟合。

    要训练模型，首先要构建一个Solver实例，将模型，数据集和各种选项（学习速率，批量大小等）传递给构造函数。
    然后调用train()方法来运行优化过程并训练模型。

    在train()方法返回之后，model.params将包含在训练过程中在验证集上表现最佳的参数。
    此外，实例变量solver.loss_history将包含训练期间遇到的所有损失的列表，实例变量
    solver.train_acc_history和solver.val_acc_history将是每个时期训练和验证集上模型精度的列表。

    Example usage might look something like this:
    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()

    A Solver works on a model object that must conform to the following API:
    - model.params must be a dictionary mapping string parameter names to numpy arrays
        containing parameter values.
    - model.loss(X,y) must be a function that computes training-time loss and gradients,and
         test-time classification scores, with the following inputs and outputs:

         Inputs:
         - X: Array giving a mini-batch of input data of shape(N, d_1, ..., d_k)
         - y: Array of labels, of shape(N,) giving labels for X where y[i] is the label for
            X[i].

         Returns:
         If y is None, run a test-time forward pass and return:
         - scores: Array of shape(N,C) giving classification scores for X where scores[i,c]
            gives the score of class c for X[i].
         If y is not None run a training time forward and backward pass and return a tuple of:
         - loss: Scalar giving the loss
         - grads: Dictionary with the same keys as self.params mapping parameter name to
            gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above.
        - data: A dictionary of training and validation data containing:
            'X_train': Array, shape(N_train, d_1, ..., d_k) of training images
            'X_val': Array, shape(N_val, d_1, ..., d_k) of validation images
            'y_train': Array, shape(N_train, d_1, ..., d_k) of labels of training images
            'y_val': Array, shape(N_val, d_1, ..., d_k) of labels of validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py. Default is'sgd'
        - optim_config: 包含超参数的字典，将传递给所选的更新规则。 每个更新规则都需要不同的超参数
            （请参阅optim.py），但所有更新规则都需要'learning_rate'参数，以便始终存在。
        - lr_decay: 学习率下降比例，每个epoch后，学习率乘以该值
        - batch_size:Size of minibatches used to compute loss and gradient during training.
        - num_epochs: 训练时的epoch数量
        - print_every: 每训练print_every次后，输出loss值
        - verbose: Boolean 设置为false时不会在训练时输出内容
        - num_train_samples: Number of training samples used to check training
            accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples:  Number of validation samples to use to check val accuracy;
            default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every epoch.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = '.'.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exist, then replace the string name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient updates. This is called by train() and should not be called
        manually
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            'model': self.model,
            'update_rule': self.update_rule,
            'lr_decay': self.lr_decay,
            'optim_config': self.optim_config,
            'batch_size': self.batch_size,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to %s' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        -------
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model on num_samples data points.
        - batch_size: Split X and y into batches of this size to avoid using too much memory.

        Returns:
        --------
        - acc: Scalar giving the fraction of instances thar were correctly classified by the model
        """
        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t+1, num_iterations, self.loss_history[-1]
                ))

            # At the end of every epoch, increment the epoch counter and decay the learnning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last iteration,
            # and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val, num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc
                    ))
                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
