import numpy as np


def _gen_names(type_name, length):
    names = list()
    for i in range(0, length):
        names.append(type_name + " " + str(i))
    return names


def _get_hat_matrix(X):
    # covariance matrix E[xx.T] = X.T @ X
    J = X.T @ X
    # Preconditioning matrix, were each row x contains: E[xx.T]^+ x
    hat = X @ np.linalg.pinv(J).T
    return hat


def _get_coef_influence(X, y, fitted_params, hat=None):
    if hat is None:
        hat = _get_hat_matrix(X)
    # vector of residuals, where each entry is (y - x.T theta)
    residuals = (y - X @ fitted_params)
    # influence on coefficients: E[xx.T]^+ row-wise-multiply(X, residual)
    influence = residuals.reshape(-1, 1) * hat
    return influence


def _get_corrected_influence(X, y, fitted_params, d_t, hat=None):
    # returns (d_t, n)
    if hat is None:
        hat = _get_hat_matrix(X)
    # influence on coefficients: E[xx.T]^+ row-wise-multiply(X, residual)
    influence_coefs = np.array(np.split(_get_coef_influence(X, y, fitted_params, hat=hat), d_t))
    treatment_X = np.array(np.split(X.T, d_t))
    corr_own_pred_influences = list()
    for i in range(0, d_t):
        # influence on predictions (n_samples, n_samples): influence_on_coefs @ X.T
        pred_influence = influence_coefs[i] @ treatment_X[i]
        # influence of sample on prediction at that sample
        own_pred_influence = np.diag(pred_influence)
        # finite sample corrected score
        corr_own_pred_influences.append(own_pred_influence / (1 - np.diag(hat @ X.T)))
    return np.array(corr_own_pred_influences)


class InfluenceResult():
    def __init__(self, influences, axis_labels, slice_labels, flatten=False):
        self.influences = influences
        self.axis_labels = axis_labels
        self.slice_labels = slice_labels
        assert (len(influences.shape) == len(axis_labels))
        for i in range(0, len(axis_labels)):
            assert(influences.shape[i] == len(slice_labels[i]))
        if flatten:
            self.flatten_influences()

    def flatten_influences(self):
        og_shape = self.influences.shape
        axis_inds = list()
        for i in range(0, len(og_shape)):
            if og_shape[i] <= 1:
                axis_inds.append(i)
        self._aggregate_dim_inds(tuple(axis_inds))

    def some_plots():
        pass

    def _aggregate_dim_inds(self, axis_inds):
        num_dims = len(self.influences.shape)
        new_axis_labels = list()
        new_slice_labels = list()
        for i in range(0, num_dims):
            if not (i in axis_inds):
                new_axis_labels.append(self.axis_labels[i])
                new_slice_labels.append(self.slice_labels[i])
        self.axis_labels = new_axis_labels
        self.slice_labels = new_slice_labels
        self.influences = np.sum(self.influences, axis_inds)

    def aggregate(self, axis_aggregates):
        axis_inds = list()
        for i in range(0, len(axis_aggregates)):
            if axis_aggregates[i] in self.axis_labels:
                axis_inds.append(self.axis_labels.index(axis_aggregates[i]))
        self._aggregate_dim_inds(tuple(axis_inds))
