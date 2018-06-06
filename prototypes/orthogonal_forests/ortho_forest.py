"""Orthogonal Random Forest

Orthogonal Random Forest (ORF) is an algorithm for heterogenous treatment effect
estimation. Orthogonal Random Forest combines orthogonalization,
a technique that effectively removes the confounding effect in two-stage estimation,
with generalized random forests, a flexible method for estimating treatment
effect heterogeneity.

This module consists of classes that implement the following variants of the ORF method:

- The ``BaseOrthoForest``, a one-forest approach for learning treatment effects that uses
  the classic RandomForest kernel to predict a treatment effect from features. A base class
  for more sophisticated ORF versions. 

- The ``DishonestOrthoForest``, a one-forest approach for learning treatment effects with
  kernel two stage estimation for predicting treatment effects. To be used as a comparison
  to the two-forest, honest version. 

- The ``OrthoForest``, a two-forest approach for learning treatment effects with
  kernel two stage estimation for predicting treatment effects.

For more details on these methods, see our paper <Orthogonal Random Forest for Heterogeneous
Treatment Effect Estimation> on arxiv.
"""

# Authors: Miruna Oprescu <moprescu@microsoft.com>
#          Vasilis Syrgkanis <vasy@microsoft.com>
#          Steven Wu <zhiww@microsoft.com>

import inspect
import numpy as np
import warnings
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LassoCV, Lasso
from residualizer import dml, second_order_dml
from causal_tree import CausalTree

def _build_tree_in_parallel(W, x, T, Y, min_leaf_size, max_splits, residualizer, model_T, model_Y):
    tree = OrthoTree(min_leaf_size=min_leaf_size, max_splits=max_splits, residualizer=residualizer,
                     model_T=model_T, model_Y=model_Y)
    tree.fit(W, x, T, Y)
    return tree


class OrthoTree(object):
    """Base tree estimator for OrthoForest classes.

    Parameters
    ----------
    min_leaf_size : integer, optional (default=20)
        The minimum number of samples in a leaf.
    
    max_splits : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree. 
    
    residualizer : class, optional (default=dml)
        The residualizer to be used at the leafs to for removing confounding effects.
        Two out of the box options provided: `dml` (double machine learning) and
        `second_order_dml` (second order double machine learning), but users can
        define a custom residualizer as long as the interface matches the `dml`
        interface.

    model_T : estimator, optional (default=sklearn.linear_model.LassoCV())
        The estimator for residualizing the treatment at the leaf. Must implement
        `fit` and `predict` methods.

    model_Y : estimator, optional (default=sklearn.linear_model.LassoCV())
        The estimator for residualizing the outcome at the leaf. Must implement
        `fit` and `predict` methods.
    """ 
    def __init__(self, min_leaf_size=20, max_splits=10, residualizer=dml,
                             model_T=LassoCV(),
                             model_Y=LassoCV()):
        self.min_leaf_size = min_leaf_size
        self.max_splits = max_splits
        self.residualizer = residualizer
        self.model_T = model_T
        self.model_Y = model_Y
    
    def fit(self, W, x, T, Y):
        """Build a causal tree from a training set (W, x, T, Y)

        Parameters
        ----------
        W : array-like, shape [n_samples, n_controls]
            High-dimensional controls.

        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.

        T : array_like, shape [n_samples]
            Treatment policy.

        Y : array_like, shape [n_samples]
            Outcome for the treatment policy.
        """
        # Initialize causal tree parameters
        self.ct = CausalTree(W, x, T, Y, self.model_T, self.model_Y, 
                             min_leaf_size=self.min_leaf_size, max_splits=self.max_splits,
                             residualizer=self.residualizer)
        # Create splits of causal tree
        self.ct.create_splits()
        # Estimate treatment effects at the leafs
        self.ct.estimate()
    
    def predict(self, x):
        """Predict treatment effects for features x.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.
        """
        # Compute heterogeneous treatement effect for x's in x_list by finding
        # the corresponding split and associating the effect computed on that leaf
        out_tau = np.zeros(x.shape[0])
        for idx, out_x in enumerate(x):
            # Find the leaf node that this x belongs too and parse the corresponding estimate
            out_tau[idx] = self.ct.find_split(out_x).estimate

        return out_tau


class BaseOrthoForest(object):
    """A one-forest approach for learning treatment effects using
        the classic RandomForest kernel for prediction. 

    Parameters
    ----------
    n_trees : integer, optional (default=10)
        Number of causal estimators in the forest.

    min_leaf_size : integer, optional (default=20)
        The minimum number of samples in a leaf.
    
    max_splits : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree. 
    
    residualizer : class, optional (default=dml)
        The residualizer to be used at the leafs to for removing confounding effects.
        Two out of the box options provided: `dml` (double machine learning) and
        `second_order_dml` (second order double machine learning), but users can
        define a custom residualizer as long as the interface matches the `dml`
        interface.

    model_T : estimator, optional (default=sklearn.linear_model.LassoCV())
        The estimator for residualizing the treatment at the leaf. Must implement
        `fit` and `predict` methods.

    model_Y : estimator, optional (default=sklearn.linear_model.LassoCV())
        The estimator for residualizing the outcome at the leaf. Must implement
        `fit` and `predict` methods.
    """ 
    def __init__(self, n_trees=10, min_leaf_size=20, max_splits=10, 
                            subsample_ratio=1.0, bootstrap=True, 
                            residualizer=dml,
                            model_T=LassoCV(),
                            model_Y=LassoCV()):
        self.n_trees = n_trees
        self.min_leaf_size = min_leaf_size
        self.max_splits = max_splits
        self.bootstrap = bootstrap
        self.subsample_ratio = subsample_ratio
        self.residualizer = residualizer
        self.model_T = model_T
        self.model_Y = model_Y
        self.subsample_ind = None

    def fit_forest(self, W, x, T, Y):
        if self.bootstrap:
            subsample_ind = np.random.choice(W.shape[0], size=(self.n_trees, W.shape[0]), replace=True)
        else:
            if self.subsample_ratio > 1.0:
                # Safety check
                self.subsample_ratio = 1.0
            subsample_size = int(self.subsample_ratio * W.shape[0])
            subsample_ind = np.zeros((self.n_trees, subsample_size))
            for t in range(self.n_trees):
                subsample_ind[t] = np.random.choice(W.shape[0], size=subsample_size, replace=False)
            subsample_ind = subsample_ind.astype(int)
        
        return subsample_ind, Parallel(n_jobs=-1, verbose=3)(
                        delayed(_build_tree_in_parallel)(
                            W[s], x[s], T[s], Y[s],
                            self.min_leaf_size, self.max_splits, self.residualizer, 
                            clone(self.model_T), clone(self.model_Y)) for s in subsample_ind)
    
    def fit(self, W, x, T, Y):
        """Build an orthogonal random forest from a training set (W, x, T, Y)

        Parameters
        ----------
        W : array-like, shape [n_samples, n_controls]
            High-dimensional controls.

        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.

        T : array_like, shape [n_samples]
            Treatment policy.

        Y : array_like, shape [n_samples]
            Outcome for the treatment policy.
        """
        self.subsample_ind, self.trees = self.fit_forest(W, x, T, Y)
    
    def predict(self, x):
        """Predict treatment effects for features x.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.
        """
        out_tau = np.zeros((x.shape[0], self.n_trees))
        for t, tree in enumerate(self.trees):
            out_tau[:, t] = tree.predict(x)
        return np.mean(out_tau, axis=1)

    def predict_interval(self, x, lower=5, upper=95):
        """Confidence intervals for prediction.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.

        lower : float, optional (default=5)
            Float between 0 and 100 representing lower percentile of confidence interval. 

        upper : float, optional (default=95)
            Float between 0 and 100 representing upper percentile of confidence interval.
            Must be larger than the lower percentile.
        """
        out_tau = np.zeros((x.shape[0], self.n_trees))
        for t, tree in enumerate(self.trees):
            out_tau[:, t] = tree.predict(x)
        return np.percentile(out_tau, lower, axis=1), np.percentile(out_tau, upper, axis=1)

    
class DishonestOrthoForest(BaseOrthoForest):
    """A one-forest approach for learning treatment effects with
        kernel two stage estimation for predicting treatment effects.

    Parameters
    ----------
    n_trees : integer, optional (default=10)
        Number of causal estimators in the forest.

    min_leaf_size : integer, optional (default=20)
        The minimum number of samples in a leaf.
    
    max_splits : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree. 

    subsample_ratio : float, optional (default=1.0)
        The ratio of the total sample to be used when training a causal tree.
        Values greater than 1.0 will be considered equal to 1.0.
        Parameter is ignored when bootstrap=True.

    bootstrap : boolean, optional (default=True)
        Whether to use bootstrap subsampling. 
    
    residualizer : class, optional (default=dml)
        The residualizer to be used at the leafs to for removing confounding effects.
        Two out of the box options provided: `dml` (double machine learning) and
        `second_order_dml` (second order double machine learning), but users can
        define a custom residualizer as long as the interface matches the `dml`
        interface.

    model_T : estimator, optional (default=sklearn.linear_model.LassoCV())
        The estimator for residualizing the treatment at the leaf. Must implement
        `fit` and `predict` methods.

    model_Y :  estimator, optional (default=sklearn.linear_model.LassoCV())
        The estimator for residualizing the outcome at the leaf. Must implement
        `fit` and `predict` methods.

    model_T_final : estimator, optional (default=None)
        The estimator for residualizing the treatment at prediction time. Must implement
        `fit` and `predict` methods. If parameter is set to `None`, it defaults to the
        value of `model_T` parameter. 

    model_Y_final : estimator, optional (default=None)
    The estimator for residualizing the outcome at prediction time. Must implement
        `fit` and `predict` methods. If parameter is set to `None`, it defaults to the
        value of `model_Y` parameter.
    """ 
    def __init__(self, n_trees=10, min_leaf_size=20, max_splits=10, 
                        subsample_ratio=1.0, bootstrap=True, 
                        residualizer=dml,
                        model_T=LassoCV(),
                        model_Y=LassoCV(),
                        model_T_final=None,
                        model_Y_final=None):
        super(DishonestOrthoForest, self).__init__(n_trees=n_trees,
                                                min_leaf_size=min_leaf_size,
                                                max_splits=max_splits,
                                                subsample_ratio=subsample_ratio, bootstrap=bootstrap,
                                                residualizer=residualizer,
                                                model_T=model_T,
                                                model_Y=model_Y)
        self.model_T_final = model_T_final
        self.model_Y_final = model_Y_final
        if self.model_T_final is None:
            self.model_T_final = clone(self.model_T)
        if self.model_Y_final is None:
            self.model_Y_final = clone(self.model_Y)
    
    def fit(self, W, x, T, Y):
        """Build an orthogonal random forest from a training set (W, x, T, Y)

        Parameters
        ----------
        W : array-like, shape [n_samples, n_controls]
            High-dimensional controls.

        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.

        T : array_like, shape [n_samples]
            Treatment policy.

        Y : array_like, shape [n_samples]
            Outcome for the treatment policy.
        """
        self.W = W
        self.T = T
        self.Y = Y
        super(DishonestOrthoForest, self).fit(W=W, x=x, T=T, Y=Y)

    def predict(self, x):
        """Predict treatment effects for features x.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.
        """
        return self._predict(x)
    
    def predict_with_weights(self, x):
        """Predict treatment effects for features x.
            Returns both treatment effects and the weights on the training points. 

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.
        """
        return self._predict(x, weights=True)

    def _predict(self, x,  weights=False):
        out_tau = np.zeros(len(x))
        model_T = self._get_weighted_pipeline(self.model_T_final)
        model_Y = self._get_weighted_pipeline(self.model_Y_final)
        for x_ind, x_out in enumerate(x):
            w, a = self._get_weights(x_out)
            mask_w = (w != 0)
            mask_a = (a != 0)
            w_nonzero = w[mask_w]
            a_nonzero = a[mask_a]
            if weights:
                if x_ind == 0:
                    x_weights = np.zeros((len(self.W_two), len(x)))
                x_weights[:, x_ind] = a
            self._fit_weighted_pipeline(model_T, self.W[mask_w], self.T[mask_w], w_nonzero)
            self._fit_weighted_pipeline(model_Y, self.W[mask_w], self.Y[mask_w], w_nonzero)
            res_T = self.T[mask_a] - model_T.predict(self.W[mask_a])
            res_Y = self.Y[mask_a] - model_Y.predict(self.W[mask_a])
            # Weighted linear regression
            out_tau[z_ind] = np.dot(res_T, a_nonzero * res_Y) / np.dot(res_T, a_nonzero * res_Y)
        if weights:
            return out_tau, np.concatenate((self.x, x_weights), axis=1)
        return out_tau

    def _get_weights(self, x_out):
        # Calculates weights
        w = np.zeros(self.W.shape[0])
        a = np.zeros(self.W.shape[0])
        for t, tree in enumerate(self.trees):
            leaf = tree.ct.find_split(x_out)
            weight_indexes = self.subsample_ind[t][leaf.est_sample_inds_1]
            leaf_weight = 1 / len(leaf.est_sample_inds_1)
            # Bootstraping has repetitions in tree sample so we need to iterate
            # over all indices
            for ind in weight_indexes:
                w[ind] += leaf_weight
            # Similar for `a` weights
            weight_indexes = self.subsample_ind[t][leaf.est_sample_inds_2]
            leaf_weight = 1 / len(leaf.est_sample_inds_2)
            for ind in weight_indexes:
                a[ind] += leaf_weight
        return (w, a)

    def _get_weighted_model(self, model_instance):
        if 'sample_weight' not in inspect.getfullargspec(model_instance.fit).args:
            # Doesn't have sample weights
            if 'sklearn.linear_model' in model_instance.__module__:
                # Is a linear model
                return ModelWrapper(model_instance, "weighted")
            else:
                return ModelWrapper(model_instance, "sampled")
        return model_instance

    def _get_weighted_pipeline(self, model_instance):
        if type(model_instance) != 'Pipeline':
            ret_model = self._get_weighted_model(model_instance)
        else:
            ret_model = clone(model_instance)
            ret_model.steps[-1] = ('weighted_model', self._get_weighted_model(model_instance.steps[-1][1]))
        return ret_model

    def _fit_weighted_pipeline(self, model_instance, x, y, weights):
        if type(model_instance) != 'Pipeline':
            model_instance.fit(x, y, weights)
        else:
            model_instance.fit(x, y, weighted_model__sample_weight=weights)


class OrthoForest(DishonestOrthoForest):
    """A two-forest approach for learning treatment effects with
        kernel two stage estimation for predicting treatment effects.

    Parameters
    ----------
    n_trees : integer, optional (default=10)
        Number of causal estimators in the forest.

    min_leaf_size : integer, optional (default=20)
        The minimum number of samples in a leaf.
    
    max_splits : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree. 

    subsample_ratio : float, optional (default=1.0)
        The ratio of the total sample to be used when training a causal tree.
        Values greater than 1.0 will be considered equal to 1.0.
        Parameter is ignored when bootstrap=True.

    bootstrap : boolean, optional (default=True)
        Whether to use bootstrap subsampling. 
    
    residualizer : class, optional (default=dml)
        The residualizer to be used at the leafs to for removing confounding effects.
        Two out of the box options provided: `dml` (double machine learning) and
        `second_order_dml` (second order double machine learning), but users can
        define a custom residualizer as long as the interface matches the `dml`
        interface.

    model_T : estimator, optional (default=sklearn.linear_model.LassoCV())
        The estimator for residualizing the treatment at the leaf. Must implement
        `fit` and `predict` methods.

    model_Y :  estimator, optional (default=sklearn.linear_model.LassoCV())
        The estimator for residualizing the outcome at the leaf. Must implement
        `fit` and `predict` methods.

    model_T_final : estimator, optional (default=None)
        The estimator for residualizing the treatment at prediction time. Must implement
        `fit` and `predict` methods. If parameter is set to `None`, it defaults to the
        value of `model_T` parameter. 

    model_Y_final : estimator, optional (default=None)
    The estimator for residualizing the outcome at prediction time. Must implement
        `fit` and `predict` methods. If parameter is set to `None`, it defaults to the
        value of `model_Y` parameter.
    """ 
    def __init__(self, n_trees=10, min_leaf_size=20, max_splits=10, 
                        subsample_ratio=1.0, bootstrap=True, 
                        residualizer=dml,
                        model_T=LassoCV(),
                        model_Y=LassoCV(),
                        model_T_final=None,
                        model_Y_final=None):
        super(OrthoForest, self).__init__(n_trees=n_trees,
                                                min_leaf_size=min_leaf_size,
                                                max_splits=max_splits,
                                                subsample_ratio=subsample_ratio, bootstrap=bootstrap,
                                                residualizer=residualizer,
                                                model_T=model_T,
                                                model_Y=model_Y,
                                                model_T_final=model_T_final,
                                                model_Y_final=model_Y_final)

    def fit(self, W, x, T, Y):
        """Build an orthogonal random forest from a training set (W, x, T, Y)

        Parameters
        ----------
        W : array-like, shape [n_samples, n_controls]
            High-dimensional controls.

        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.

        T : array_like, shape [n_samples]
            Treatment policy.

        Y : array_like, shape [n_samples]
            Outcome for the treatment policy.
        """
        n = int(W.shape[0]/2)
        self.W_one = W[:n]
        self.W_two = W[n:]
        self.T_one = T[:n]
        self.T_two = T[n:]
        self.Y_one = Y[:n]
        self.Y_two = Y[n:]
        self.x_one = x[:n]
        self.x_two = x[n:]
        self.forest_one_subsample_ind, self.forest_one_trees = self.fit_forest(W=self.W_one, x=self.x_one, T=self.T_one, Y=self.Y_one)
        self.forest_two_subsample_ind, self.forest_two_trees = self.fit_forest(W=self.W_two, x=self.x_two, T=self.T_two, Y=self.Y_two)

    def predict(self, x):
        """Predict treatment effects for features x.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.
        """
        return self._predict(x)
    
    def predict_with_weights(self, x):
        """Predict treatment effects for features x.
            Returns both treatment effects and the weights on the training points. 

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Feature vector that captures heterogeneity.
        """
        return self._predict(x, weights=True)

    def _predict(self, x, weights=False):
        out_tau = np.zeros(len(x))
        model_T = self._get_weighted_pipeline(self.model_T_final)
        model_Y = self._get_weighted_pipeline(self.model_Y_final)
        for x_ind, x_out in enumerate(x):
            w, a = self._get_weights(x_out)
            mask_w = (w != 0)
            mask_a = (a != 0)
            w_nonzero = w[mask_w]
            a_nonzero = a[mask_a]
            if weights:
                if x_ind == 0:
                    x_weights = np.zeros((len(self.x_two), len(x)))
                x_weights[:, x_ind] = a
            self._fit_weighted_pipeline(model_T, self.W_one[mask_w], self.T_one[mask_w], w_nonzero)
            self._fit_weighted_pipeline(model_Y, self.W_one[mask_w], self.Y_one[mask_w], w_nonzero)
            res_T = self.T_two[mask_a] - model_T.predict(self.W_two[mask_a])
            res_Y = self.Y_two[mask_a] - model_Y.predict(self.W_two[mask_a])
            # Weighted linear regression
            out_tau[x_ind] = np.dot(res_T, a_nonzero * res_Y) / np.dot(res_T, a_nonzero * res_T)
        if weights:
            return out_tau, np.concatenate((self.x_two, x_weights), axis=1)
        return out_tau

    def _get_weights(self, x_out):
        # Calculates weights
        w = np.zeros(self.W_one.shape[0])
        a = np.zeros(self.W_two.shape[0])
        for t, tree in enumerate(self.forest_one_trees):
            leaf = tree.ct.find_split(x_out)
            weight_indexes = self.forest_one_subsample_ind[t][leaf.est_sample_inds]
            leaf_weight = 1 / len(leaf.est_sample_inds)
            # Bootstraping has repetitions in tree sample so we need to iterate
            # over all indices
            for ind in weight_indexes:
                w[ind] += leaf_weight
        for t, tree in enumerate(self.forest_two_trees):
            leaf = tree.ct.find_split(x_out)
            # Similar for `a` weights
            weight_indexes = self.forest_two_subsample_ind[t][leaf.est_sample_inds]
            leaf_weight = 1 / len(leaf.est_sample_inds)
            for ind in weight_indexes:
                a[ind] += leaf_weight
        return (w, a)


class ModelWrapper(object):
    """Helper class for assiging weights to models without this option.

        Parameters
        ----------
        model_instance : estimator
            Model that requires weights.

        sample_type : string, optional (default=`weighted`)
            Method for adding weights to the model. `wighted` for linear models
            where the weights can be incorporated in the matrix multiplication,
            `sampled` for other models. `sampled` samples the training set according
            to the normalized weights and creates a dataset larger than the original.

        """
    def __init__(self, model_instance, sample_type="weighted"):
        self.model_instance = model_instance
        if sample_type == "weighted":
            self.data_transform = self._weighted_inputs
        else:
            warnings.warn("The model provided does not support sample weights. " +  
                        "Manual weighted sampling may icrease the variance in the results.")
            self.data_transform = self._sampled_inputs

    def __getattr__(self, name):
        func = getattr(self.__dict__['model_instance'], name)
        if name == "fit":
            return self._fit(func)
        else:
            return func
    
    def _fit(self, func):
        def fit(X, y, sample_weight=None):
            if sample_weight is None:
                return func(X, y)
            X, y = self.data_transform(X, y, sample_weight)
            return func(X, y)
        return fit
        
    def _weighted_inputs(self, X, y, sample_weight):
        normalized_weights = X.shape[0] * sample_weight / np.sum(sample_weight)
        sqrt_weights = np.sqrt(normalized_weights)
        return (X.T * sqrt_weights).T, sqrt_weights * y
    
    def _sampled_inputs(self, X, y, sample_weight):
        # normalize weights
        normalized_weights = sample_weight / np.sum(sample_weight)
        data_length = int(min(1 / np.min(normalized_weights[normalized_weights > 0]), 10) * X.shape[0])
        data_indices = np.random.choice(X.shape[0], size=data_length, p=normalized_weights)
        return X[data_indices], y[data_indices]    