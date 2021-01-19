# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Basic tree utilities and methods.

Class :class:`CausalTree` is the base estimator for the Orthogonal Random Forest, whereas
class :class:`Node` represents the core unit of the :class:`CausalTree` class.
"""

import numpy as np
from sklearn.utils import check_random_state


class Node:
    """Building block of :class:`CausalTree` class.

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
        self.left = None
        self.right = None

    def find_tree_node(self, value):
        """
        Recursively find and return the node of the causal tree that corresponds
        to the input feature vector.

        Parameters
        ----------
        value : array-like, shape (d_x,)
            Feature vector whose node we want to find.
        """
        if self.feature == -1:
            return self
        elif value[self.feature] < self.threshold:
            return self.left.find_tree_node(value)
        else:
            return self.right.find_tree_node(value)


class CausalTree:
    """Base class for growing an OrthoForest.

    Parameters
    ----------
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

    max_depth : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree.

    n_proposals :  int, optional (default=1000)
        Number of split proposals to be considered. A smaller number will improve
        execution time, but might reduce accuracy of prediction.

    balancedness_tol : float, optional (default=.3)
        Tolerance for balance between child nodes in a split. A smaller value
        will result in an unbalanced tree prone to overfitting. Has to lie
        between 0 and .5 as it is used to control both directions of imbalancedness.
        With the default value we guarantee that each child of a split contains
        at least 20% and at most 80% of the data of the parent node.

    random_state : int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self,
                 min_leaf_size=10,
                 max_depth=10,
                 n_proposals=1000,
                 balancedness_tol=.3,
                 random_state=None):
        # Causal tree parameters
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.balancedness_tol = balancedness_tol
        self.n_proposals = n_proposals
        self.random_state = check_random_state(random_state)
        # Tree structure
        self.tree = None

    def create_splits(self, Y, T, X, W,
                      nuisance_estimator, parameter_estimator, moment_and_mean_gradient_estimator):
        """
        Recursively build a causal tree.

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
        """
        # No need for a random split since the data is already
        # a random subsample from the original input
        n = Y.shape[0] // 2
        self.tree = Node(np.arange(n), np.arange(n, Y.shape[0]))
        # node list stores the nodes that are yet to be splitted
        node_list = [(self.tree, 0)]

        while len(node_list) > 0:
            node, depth = node_list.pop()

            # If by splitting we have too small leaves or if we reached the maximum number of splits we stop
            if node.split_sample_inds.shape[0] // 2 >= self.min_leaf_size and depth < self.max_depth:

                # Create local sample set
                node_X = X[node.split_sample_inds]
                node_W = W[node.split_sample_inds] if W is not None else None
                node_T = T[node.split_sample_inds]
                node_Y = Y[node.split_sample_inds]
                node_X_estimate = X[node.est_sample_inds]
                node_size_split = node_X.shape[0]
                node_size_est = node_X_estimate.shape[0]

                # Compute nuisance estimates for the current node
                nuisance_estimates = nuisance_estimator(node_Y, node_T, node_X, node_W)
                if nuisance_estimates is None:
                    # Nuisance estimate cannot be calculated
                    continue
                # Estimate parameter for current node
                node_estimate = parameter_estimator(node_Y, node_T, node_X, nuisance_estimates)
                if node_estimate is None:
                    # Node estimate cannot be calculated
                    continue
                # Calculate moments and gradient of moments for current data
                moments, mean_grad = moment_and_mean_gradient_estimator(
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
                        continue
                    else:
                        raise exc
                # Calculate point-wise pseudo-outcomes rho
                rho = np.matmul(moments, inverse_grad)

                # a split is determined by a feature and a sample pair
                # the number of possible splits is at most (number of features) * (number of node samples)
                n_proposals = min(self.n_proposals, node_X.size)
                #  we draw random such pairs by drawing a random number in {0, n_feats * n_node_samples}
                random_pair = self.random_state.choice(node_X.size, size=n_proposals, replace=False)
                # parse row and column of random pair
                thr_inds, dim_proposals = np.unravel_index(random_pair, node_X.shape)
                # the sample of the pair is the integer division of the random number with n_feats
                thr_proposals = node_X[thr_inds, dim_proposals]

                # calculate the binary indicator of whether sample i is on the left or the right
                # side of proposed split j. So this is an n_samples x n_proposals matrix
                side = node_X[:, dim_proposals] < thr_proposals
                # calculate the number of samples on the left child for each proposed split
                size_left = np.sum(side, axis=0)
                # calculate the analogous binary indicator for the samples in the estimation set
                side_est = node_X_estimate[:, dim_proposals] < thr_proposals
                # calculate the number of estimation samples on the left child of each proposed split
                size_est_left = np.sum(side_est, axis=0)

                # find the upper and lower bound on the size of the left split for the split
                # to be valid so as for the split to be balanced and leave at least min_leaf_size
                # on each side.
                lower_bound = max((.5 - self.balancedness_tol) * node_size_split, self.min_leaf_size)
                upper_bound = min((.5 + self.balancedness_tol) * node_size_split, node_size_split - self.min_leaf_size)
                valid_split = (lower_bound <= size_left)
                valid_split &= (size_left <= upper_bound)

                # similarly for the estimation sample set
                lower_bound_est = max((.5 - self.balancedness_tol) * node_size_est, self.min_leaf_size)
                upper_bound_est = min((.5 + self.balancedness_tol) * node_size_est, node_size_est - self.min_leaf_size)
                valid_split &= (lower_bound_est <= size_est_left)
                valid_split &= (size_est_left <= upper_bound_est)

                # if there is no valid split then don't create any children
                if ~np.any(valid_split):
                    continue

                # filter only the valid splits
                valid_dim_proposals = dim_proposals[valid_split]
                valid_thr_proposals = thr_proposals[valid_split]
                valid_side = side[:, valid_split]
                valid_size_left = size_left[valid_split]
                valid_side_est = side_est[:, valid_split]

                # calculate the average influence vector of the samples in the left child
                left_diff = np.matmul(rho.T, valid_side)
                # calculate the average influence vector of the samples in the right child
                right_diff = np.matmul(rho.T, 1 - valid_side)
                # take the square of each of the entries of the influence vectors and normalize
                # by size of each child
                left_score = left_diff**2 / valid_size_left.reshape(1, -1)
                right_score = right_diff**2 / (node_size_split - valid_size_left).reshape(1, -1)
                # calculate the vector score of each candidate split as the average of left and right
                # influence vectors
                spl_score = (right_score + left_score) / 2

                # eta specifies how much weight to put on individual heterogeneity vs common heterogeneity
                # across parameters. we give some benefit to individual heterogeneity factors for cases
                # where there might be large discontinuities in some parameter as the conditioning set varies
                eta = np.random.uniform(0.25, 1)
                # calculate the scalar score of each split by aggregating across the vector of scores
                split_scores = np.max(spl_score, axis=0) * eta + np.mean(spl_score, axis=0) * (1 - eta)

                # Find split that minimizes criterion
                best_split_ind = np.argmax(split_scores)
                node.feature = valid_dim_proposals[best_split_ind]
                node.threshold = valid_thr_proposals[best_split_ind]

                # Create child nodes with corresponding subsamples
                left_split_sample_inds = node.split_sample_inds[valid_side[:, best_split_ind]]
                left_est_sample_inds = node.est_sample_inds[valid_side_est[:, best_split_ind]]
                node.left = Node(left_split_sample_inds, left_est_sample_inds)
                right_split_sample_inds = node.split_sample_inds[~valid_side[:, best_split_ind]]
                right_est_sample_inds = node.est_sample_inds[~valid_side_est[:, best_split_ind]]
                node.right = Node(right_split_sample_inds, right_est_sample_inds)

                # add the created children to the list of not yet split nodes
                node_list.append((node.left, depth + 1))
                node_list.append((node.right, depth + 1))

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

    def find_split(self, value):
        return self.tree.find_tree_node(value.astype(np.float64))
