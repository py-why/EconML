# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Provides linear regressors with support for applying L1 and/or L2 regularization to a subset of coefficients."""

from warnings import warn
import numpy as np
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class SelectiveRegularization:
    """
    Estimator of a linear model where regularization is applied to only a subset of the coefficients.

    Assume that our loss is

    .. math::
        \\ell(\\beta_1, \\beta_2) = \\lVert y - X_1 \\beta_1 - X_2 \\beta_2 \\rVert^2 + f(\\beta_2)

    so that we're regularizing only the coefficients in :math:`\\beta_2`.

    Then, since :math:`\\beta_1` doesn't appear in the penalty, the problem of finding :math:`\\beta_1` to minimize the
    loss once :math:`\\beta_2` is known reduces to just a normal OLS regression, so that:

    .. math::
        \\beta_1 = (X_1^\\top X_1)^{-1}X_1^\\top(y - X_2 \\beta_2)

    Plugging this into the loss, we obtain

    .. math::

        ~& \\lVert y - X_1 (X_1^\\top X_1)^{-1}X_1^\\top(y - X_2 \\beta_2) - X_2 \\beta_2 \\rVert^2 + f(\\beta_2) \\\\
        =~& \\lVert (I - X_1 (X_1^\\top  X_1)^{-1}X_1^\\top)(y - X_2 \\beta_2) \\rVert^2 + f(\\beta_2)

    But, letting :math:`M_{X_1} = I - X_1 (X_1^\\top  X_1)^{-1}X_1^\\top`, we see that this is

    .. math::
        \\lVert (M_{X_1} y) - (M_{X_1} X_2) \\beta_2 \\rVert^2 + f(\\beta_2)

    so finding the minimizing :math:`\\beta_2` can be done by regressing :math:`M_{X_1} y` on :math:`M_{X_1} X_2` using
    the penalized regression method incorporating :math:`f`.  Note that these are just the residual values of :math:`y`
    and :math:`X_2` when regressed on :math:`X_1` using OLS.

    Parameters
    ----------
    unpenalized_inds : list of int
        The indices that should not be penalized when the model is fit; all other indices will be penalized
    penalized_model : :term:`regressor`
        A penalized linear regression model
    fit_intercept : bool, optional, default True
        Whether to fit an intercept; the intercept will not be penalized if it is fit
    """

    def __init__(self, unpenalized_inds, penalized_model, fit_intercept=True):
        self._unpenalized_inds = unpenalized_inds
        self._penalized_model = clone(penalized_model)
        self._fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like, shape (n, d_x)
            The features to regress against
        y : array-like, shape (n,) or (n, d_y)
            The regression target
        sample_weight : array-like, shape (n,), optional, default None
            Relative weights for each sample
        """
        self._penalized_inds = np.delete(np.arange(X.shape[1]), self._unpenalized_inds)
        X1 = X[:, self._unpenalized_inds]
        X2 = X[:, self._penalized_inds]

        X2_res = X2 - LinearRegression(fit_intercept=self._fit_intercept).fit(X1, X2,
                                                                              sample_weight=sample_weight).predict(X1)
        y_res = y - LinearRegression(fit_intercept=self._fit_intercept).fit(X1, y,
                                                                            sample_weight=sample_weight).predict(X1)

        if sample_weight is not None:
            self._penalized_model.fit(X2_res, y_res, sample_weight=sample_weight)
        else:
            self._penalized_model.fit(X2_res, y_res)

        # The unpenalized model can't contain an intercept, because in the analysis above
        # we rely on the fact that M(X beta) = (M X) beta, but M(X beta + c) is not the same
        # as (M X) beta + c, so the learned coef and intercept will be wrong
        intercept = self._penalized_model.predict(np.zeros_like(X2[0:1]))
        if not np.allclose(intercept, 0):
            raise AttributeError("The penalized model has a non-zero intercept; to fit an intercept "
                                 "you should instead either set fit_intercept to True when initializing the "
                                 "SelectiveRegression instance (for an unpenalized intercept) or "
                                 "explicitly add a column of ones to the data being fit and include that "
                                 "column in the penalized indices.")

        # now regress X1 on y - X2 * beta2 to learn beta1
        self._model_X1 = LinearRegression(fit_intercept=self._fit_intercept)
        self._model_X1.fit(X1, y - self._penalized_model.predict(X2), sample_weight=sample_weight)

        return self

    def predict(self, X):
        """
        Make a prediction for each sample.

        Parameters
        ----------
        X : array-like, shape (m, d_x)
            The samples whose targets to predict

        Output
        ------
        arr : array-like, shape (m,) or (m, d_y)
            The predicted targets
        """
        X1 = X[:, self._unpenalized_inds]
        X2 = X[:, self._penalized_inds]
        return self._model_X1.predict(X1) + self._penalized_model.predict(X2)

    @property
    def coef_(self):
        """
        Get the coefficient matrix of the predictor.

        Output
        ------
        arr : array, shape (d_x) or (d_y,d_x)
        """
        y_dim = self._model_X1.coef_.shape[0:-1]  # get () if y was a vector, or (d_y,) otherwise
        x_dim = len(self._penalized_inds) + len(self._unpenalized_inds),
        coefs = np.empty(y_dim + x_dim)
        coefs[..., self._penalized_inds] = self._penalized_model.coef_
        coefs[..., self._unpenalized_inds] = self._model_X1.coef_
        return coefs

    @property
    def intercept_(self):
        """
        Get the intercept of the predictor.

        Output
        ------
        arr : scalar or array of shape (d_y,)
        """
        # Note that the penalized model should *not* have an intercept
        return self._model_X1.intercept_

    def score(self, X, y):
        """
        Score the predictions for a set of features to ground truth.

        Parameters
        ----------
        X : array-like, shape (m, d_x)
            The samples to predict
        y : array-like, shape (m,) or (m, d_y)
            The ground truth targets

        Output
        ------
        score : float
            The model's score
        """
        return r2_score(y, self.predict(X))
