# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Basic tree utilities and methods.

Class `CausalTree` is the base estimator for the Orthogonal Random Forest, whereas
class `Node` represents the core unit of the `CausalTree` class.
"""

import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import scipy.special


class Node:
    """Building block of `CausalTree` class.

    Parameters
    ----------
    sample_inds : array-like, shape (n, )
        Indices defining the sample that the split criterion will be computed on.

    estimate_inds : array-like, shape (n, )
        Indices defining the sample used for calculating balance criteria.

    """

    def __init__(self, sample_inds, estimate_inds):
        self.feature = -1
        self.threshold = np.inf
        self.split_sample_inds = sample_inds
        self.est_sample_inds = estimate_inds
        self.estimate = None
        self.left = None
        self.right = None


class CausalTree:
    """Base class for growing an `OrthoTree`.

    Parameters
    ----------
    Y : array-like, shape (n, d_y)
            Outcome for the treatment policy.

    T : array-like, shape (n, d_t)
        Treatment policy.

    X : array-like, shape (n, d_x)
        Feature vector that captures heterogeneity.

    W : array-like, shape (n, d_w) or None (default=None)
        High-dimensional controls.

    nuisance_estimator : method
        Method that estimates the nuisances at each node.
        Takes in (Y, T, X, W) and returns nuisance estimates.

    parameter_estimator : method
        Method that estimates the parameter of interest at each node.
        Takes in (Y, T, nuisance_estimates) and returns the parameter estimate.

    moment_and_mean_gradient_estimator : method
        Method that estimates the moments and mean moment gradient at each node.
        Takes in (Y, T, X, W, nuisance_estimates, parameter_estimate) and returns
        the moments and the mean moment gradient.

    min_leaf_size : integer, optional (default=10)
        The minimum number of samples in a leaf.

    max_splits : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree.

    n_proposals :  int, optional (default=1000)
        Number of split proposals to be considered. A smaller number will improve
        execution time, but might reduce accuracy of prediction.

    balancedness_tol : float, optional (default=.3)
        Tolerance for balance between child nodes in a split. A smaller value
        will result in an unbalanced tree prone to overfitting.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self,
                 Y, T, X, W,
                 nuisance_estimator,
                 parameter_estimator,
                 moment_and_mean_gradient_estimator,
                 min_leaf_size=10,
                 max_splits=10,
                 n_proposals=1000,
                 balancedness_tol=.3,
                 random_state=None):
        # Input datasets
        self.W = W
        self.X = X
        self.T = T
        self.Y = Y
        # Estimators
        self.nuisance_estimator = nuisance_estimator
        self.parameter_estimator = parameter_estimator
        self.moment_and_mean_gradient_estimator = moment_and_mean_gradient_estimator
        # Causal tree parameters
        self.min_leaf_size = min_leaf_size
        self.max_splits = max_splits
        self.balancedness_tol = balancedness_tol
        self.n_proposals = n_proposals
        self.random_state = check_random_state(random_state)
        # Tree structure
        self.tree = None

    def recursive_split(self, node, split_acc):
        """
        Recursively build a causal tree.

        Parameters
        ----------
        node : Instance of class `Node`
            Parent node from which to recursively grow a sub-tree.

        split_acc : int
            Accumulator that counts the number of splits made up
            until this node.
        """
        # If by splitting we have too small leaves or if we reached the maximum number of splits we stop
        if node.split_sample_inds.shape[0] // 2 < self.min_leaf_size or split_acc >= self.max_splits:
            return node
        else:

            # Create local sample set
            node_X = self.X[node.split_sample_inds]
            node_W = self.W[node.split_sample_inds] if self.W is not None else None
            node_T = self.T[node.split_sample_inds]
            node_Y = self.Y[node.split_sample_inds]
            node_X_estimate = self.X[node.est_sample_inds]
            node_size_split = node_X.shape[0]
            node_size_est = node_X_estimate.shape[0]

            # Compute nuisance estimates for the current node
            nuisance_estimates = self.nuisance_estimator(node_Y, node_T, node_X, node_W)
            if nuisance_estimates is None:
                # Nuisance estimate cannot be calculated
                return
            # Estimate parameter for current node
            node_estimate = self.parameter_estimator(node_Y, node_T, node_X, nuisance_estimates)
            if node_estimate is None:
                # Node estimate cannot be calculated
                return
            # Calculate moments and gradient of moments for current data
            moments, mean_grad = self.moment_and_mean_gradient_estimator(
                node_Y, node_T, node_X, node_W,
                nuisance_estimates,
                node_estimate)
            # Calculate inverse gradient
            try:
                inverse_grad = np.linalg.inv(mean_grad)
            except np.linalg.LinAlgError as exc:
                if 'Singular matrix' in str(exc):
                    # The gradient matrix is not invertible.
                    # No good split can be found
                    return
                else:
                    raise exc
            # Calculate point-wise pseudo-outcomes rho
            rho = np.matmul(moments, inverse_grad)

            # Generate random proposals of dimensions to split
            dim_proposals = self.random_state.choice(
                np.arange(node_X.shape[1]), size=self.n_proposals, replace=True)
            proposals = []
            for t, dim in enumerate(dim_proposals):
                # Append to the proposals a tuple (dimension, threshold) where the threshold is randomly chosen
                proposals.append((dim, self.random_state.choice(np.unique(node_X[:, dim]), size=1, replace=False)[0]))
            # Compute criterion for each proposal
            split_scores = np.zeros(len(proposals))
            # eta specifies how much weight to put on individual heterogeneity vs common heterogeneity
            # across parameters. we give some benefit to individual heterogeneity factors for cases
            # where there might be large discontinuities in some parameter as the conditioning set varies
            eta = np.random.uniform(0.25, 1)
            for idx, (prop_feat, prop_thr) in enumerate(proposals):
                # If splitting creates valid leafs in terms of mean leaf size
                split_ratio_split = np.sum(
                    node_X[:, prop_feat] < prop_thr) / node_size_split
                min_leaf_size_split = min(
                    split_ratio_split * node_size_split, (1 - split_ratio_split) * node_size_split)
                split_ratio_estimate = np.sum(
                    node_X_estimate[:, prop_feat] < prop_thr) / node_size_est
                min_leaf_size_est = min(
                    split_ratio_estimate * node_size_est, (1 - split_ratio_estimate) * node_size_est)
                if min(min_leaf_size_split, min_leaf_size_est) > self.min_leaf_size and \
                   min(split_ratio_split, 1 - split_ratio_split) >= .5 - self.balancedness_tol and \
                   min(split_ratio_estimate, 1 - split_ratio_estimate) >= .5 - self.balancedness_tol:
                    # Calculate criterion for split
                    left_ind = node_X[:, prop_feat] < prop_thr
                    right_ind = node_X[:, prop_feat] >= prop_thr
                    rho_left = rho[left_ind]
                    rho_right = rho[right_ind]
                    left_diff = np.sum(rho_left, axis=0)
                    left_score = left_diff ** 2 / rho_left.shape[0]
                    right_diff = np.sum(rho_right, axis=0)
                    right_score = right_diff ** 2 / rho_right.shape[0]
                    spl_score = (right_score + left_score) / 2
                    split_scores[idx] = np.max(spl_score) * eta + np.mean(spl_score) * (1 - eta)
                else:
                    # Else set criterion to infinity so that this split is not chosen
                    split_scores[idx] = - np.inf

            # If no good split was found
            if np.max(split_scores) == - np.inf:
                return node

            # Find split that minimizes criterion
            best_split_ind = np.argmax(split_scores)
            best_split_feat, best_split_thr = proposals[best_split_ind]

            # Set the split attributes at the node
            node.feature = best_split_feat
            node.threshold = best_split_thr

            # Create child nodes with corresponding subsamples
            left_split_sample_inds = node.split_sample_inds[node_X[:, best_split_feat] < best_split_thr]
            left_est_sample_inds = node.est_sample_inds[node_X_estimate[:, best_split_feat] < best_split_thr]
            node.left = Node(left_split_sample_inds, left_est_sample_inds)
            right_split_sample_inds = node.split_sample_inds[node_X[:, best_split_feat] >= best_split_thr]
            right_est_sample_inds = node.est_sample_inds[node_X_estimate[:, best_split_feat] >= best_split_thr]
            node.right = Node(right_split_sample_inds, right_est_sample_inds)

            # Recursively split children
            self.recursive_split(node.left, split_acc + 1)
            self.recursive_split(node.right, split_acc + 1)

            # Return parent node
            return node

    def create_splits(self):
        # No need for a random split since the data is already
        # a random subsample from the original input
        n = self.Y.shape[0] // 2
        root = Node(np.arange(n), np.arange(n, self.Y.shape[0]))
        self.tree = self.recursive_split(root, 0)

    def estimate_leafs(self, node):
        if node.left or node.right:
            self.estimate_leafs(node.left)
            self.estimate_leafs(node.right)
        else:
            # Estimate the local parameter at the leaf using the estimate data
            nuisance_estimates = self.nuisance_estimator(
                self.Y[node.est_sample_inds],
                self.T[node.est_sample_inds],
                self.X[node.est_sample_inds],
                self.W[node.est_sample_inds] if self.W is not None else None)
            if nuisance_estimates is None:
                node.estimate = None
                return
            node.estimate = self.parameter_estimator(
                self.Y[node.est_sample_inds],
                self.T[node.est_sample_inds],
                self.X[node.est_sample_inds],
                nuisance_estimates)

    def estimate(self):
        self.estimate_leafs(self.tree)

    def print_tree_rec(self, node):
        if not node:
            return
        print("Node: ({}, {})".format(node.feature, node.threshold))
        print("Left Child")
        self.print_tree_rec(node.left)
        print("Right Child")
        self.print_tree_rec(node.right)

    def print_tree(self):
        self.print_tree_rec(self.tree)

    def find_tree_node(self, node, value):
        if node.feature == -1:
            return node
        elif value[node.feature] < node.threshold:
            return self.find_tree_node(node.left, value)
        else:
            return self.find_tree_node(node.right, value)

    def find_split(self, value):
        return self.find_tree_node(self.tree, value)
