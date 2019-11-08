# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Provides linear regressors with support for applying L1 and/or L2 regularization to a subset of coefficients."""

import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from itertools import product


class SelectiveElasticNet:
    """
    Estimator that allows L1 and L2 penalties on a subset of the features of a linear model.

    Parameters
    ----------
    num_outcomes : int
        The dimension of the output

    num_features : int
        The dimension of the input

    subset : set of int
        The set of feature indices to penalize

    steps : int, optional (default 1000)
        The number of iterations to run

    alpha_l1 : float, optional (default 0.1)
        The L1 penalty to apply to coefficients

    alpha_l2 : float, optional (default 0.1)
        The L2 penalty to apply to coefficients

    learning_rate : float, optional (default 0.1)
        The learning rate to use with Adagrad

    Returns
    -------
    The estimator
    """

    # TODO: allow different subsets for L1 and L2 regularization?

    def __init__(self, num_outcomes, num_features, subset, steps=1000, alpha_l1=0.1, alpha_l2=0.1, learning_rate=0.1):
        self._subset = subset
        self._subset_c = np.setdiff1d(np.arange(num_features), subset)
        self._steps = steps
        self._alpha_l1 = alpha_l1
        self._alpha_l2 = alpha_l2
        self._learning_rate = learning_rate
        self.tf_graph_init(num_outcomes, len(self._subset), len(self._subset_c))

    def tf_graph_init(self, num_outcomes, num_reg_features, num_ureg_features):
        """
        Create the graph that corresponds to the squared loss with an L1/L2 penalties.

        The penalties apply only on the subset of features specified by the self._subset variable.
        Also creates the optimizer that minimizes this loss and a persistent tensorflow
        session for the class.
        """
        self.Y = tf.placeholder("float", [None, num_outcomes], name="outcome")
        self.X_reg = tf.placeholder("float", [None, num_reg_features], name="reg_features")
        self.X_ureg = tf.placeholder("float", [None, num_ureg_features], name="ureg_features")

        self.weights_reg = tf.Variable(tf.random_normal(
            [num_reg_features, num_outcomes], 0, 0.1), name="weights_reg")
        self.weights_ureg = tf.Variable(tf.random_normal(
            [num_ureg_features, num_outcomes], 0, 0.1), name="weights_ureg")
        self.bias = tf.Variable(tf.random_normal(
            [num_outcomes], 0, 0.1), name="biases")
        self.Y_pred = tf.add(tf.add(tf.matmul(self.X_reg, self.weights_reg),
                                    tf.matmul(self.X_ureg, self.weights_ureg)), self.bias)

        regularization = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l1_l2_regularizer(
                scale_l1=self._alpha_l1, scale_l2=self._alpha_l2), [self.weights_reg])
        self.cost = tf.reduce_mean(tf.pow(self.Y - self.Y_pred, 2)) + regularization

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        self.session = tf.Session()

    def fit(self, X, y):
        """Fit the model."""
        # TODO: any better way to deal with sparsity?
        if sp.issparse(X):
            X = X.toarray()
        self.session.run(tf.global_variables_initializer())
        for step in range(self._steps):
            self.session.run(self.train, feed_dict={
                self.X_reg: X[:, self._subset],
                self.X_ureg: X[:, self._subset_c],
                self.Y: y.reshape(-1, 1)
            })
        return self

    def predict(self, X):
        """Apply the model to a set of features to predict the outcomes."""
        # TODO: any better way to deal with sparsity?
        if sp.issparse(X):
            X = X.toarray()
        return self.session.run(self.Y_pred, feed_dict={
            self.X_reg: X[:, self._subset],
            self.X_ureg: X[:, self._subset_c]
        })

    @property
    def coef_(self):
        """Get the model coefficients."""
        coef_reg = self.session.run(self.weights_reg.value())
        coef_ureg = self.session.run(self.weights_ureg.value())
        full_coef = np.zeros((coef_reg.shape[0] + coef_ureg.shape[0], coef_reg.shape[1]))
        full_coef[self._subset, :] = coef_reg
        full_coef[self._subset_c, :] = coef_ureg
        return full_coef

    def score(self, X, y):
        """Score the predictions for a set of features to ground truth."""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X).reshape(y.shape))


class SelectiveLasso(SelectiveElasticNet):
    """
    Estimator that allows L1 penalties on a subset of the features of a linear model.

    Parameters
    ----------
    num_outcomes : int
        The dimension of the output

    num_features : int
        The dimension of the input

    subset : set of int
        The set of feature indices to penalize

    steps : int, optional (default 1000)
        The number of iterations to run

    alpha_l1 : float, optional (default 0.1)
        The L1 penalty to apply to coefficients

    learning_rate : float, optional (default 0.1)
        The learning rate to use with Adagrad

    Returns
    -------
    The estimator
    """

    def __init__(self, num_outcomes, num_features, subset, steps=1000, alpha=0.1, learning_rate=0.1):
        super().__init__(num_outcomes, num_features, subset, steps=1000,
                         alpha_l1=alpha, alpha_l2=0.0, learning_rate=0.1)


class SelectiveRidge(SelectiveElasticNet):
    """
    Estimator that allows L2 penalties on a subset of the features of a linear model.

    Parameters
    ----------
    num_outcomes : int
        The dimension of the output

    num_features : int
        The dimension of the input

    subset : set of int
        The set of feature indices to penalize

    steps : int, optional (default 1000)
        The number of iterations to run

    alpha_l1 : float, optional (default 0.1)
        The L1 penalty to apply to coefficients

    alpha_l2 : float, optional (default 0.1)
        The L2 penalty to apply to coefficients

    learning_rate : float, optional (default 0.1)
        The learning rate to use with Adagrad

    Returns
    -------
    The estimator
    """

    def __init__(self, num_outcomes, num_features, subset, steps=1000, alpha=0.1, learning_rate=0.1):
        super().__init__(num_outcomes, num_features, subset, steps=1000,
                         alpha_l1=0.0, alpha_l2=alpha, learning_rate=0.1)
