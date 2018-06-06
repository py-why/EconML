"""Basic tree utilities and methods. 

Class `CausalTree` is the base estimator for the Orthogonal Random Forest, whereas
class `Node` represents the core unit of the `CausalTree` class. 
"""

import numpy as np
from sklearn.model_selection import train_test_split 
from residualizer import dml, second_order_dml

class Node:

    def __init__(self, sample_inds, estimate_inds):
        self.feature = -1
        self.threshold = np.inf
        self.split_sample_inds = sample_inds
        self.est_sample_inds = estimate_inds
        self.estimate = 0
        self.est_sample_inds_1 = None
        self.est_sample_inds_2 = None
        self.left = None
        self.right = None


class CausalTree:

    def __init__(self, W, x, T, Y, model_T, model_Y, min_leaf_size=20, max_splits=10, n_proposals=1000,
                 residualizer=dml, balancedness_tol=.3):
        # split the data into two parts: one for splitting, the other for estimation at the leafs
        self.W = W
        self.x = x
        self.T = T
        self.Y = Y

        self.model_T = model_T
        self.model_Y = model_Y
        self.tree = None
        self.min_leaf_size = min_leaf_size
        self.max_splits = max_splits
        self.balancedness_tol = balancedness_tol
        self.n_proposals = n_proposals
        self.residualizer = residualizer

    def recursive_split(self, node, split_acc):

        # If by splitting we have too small leaves or if we reached the maximum number of splits we stop
        if node.split_sample_inds.shape[0] // 2 < self.min_leaf_size or split_acc >= self.max_splits:
            return node
        else:

            # Create local sample set
            node_x = self.x[node.split_sample_inds, :]
            node_W = self.W[node.split_sample_inds, :]
            node_T = self.T[node.split_sample_inds]
            node_Y = self.Y[node.split_sample_inds]
            node_x_estimate = self.x[node.est_sample_inds]

            # compute the base estimate for the current node using double ml or second order double ml
            node.estimate, res_T, res_Y = self.residualizer(node_W, node_T, node_Y,
                                                            model_T=self.model_T, model_Y=self.model_Y)

            # compute the influence functions here that are used for the criterion
            grad = - (res_T ** 2).mean()
            moment = (res_Y - node.estimate * res_T) * res_T
            rho = - moment / grad  # TODO: Watch out for division by zero!

            # generate random proposals of dimensions to split
            dim_proposals = np.random.choice(
                np.arange(node_x.shape[1]), size=self.n_proposals, replace=True)
            proposals = []
            for t, dim in enumerate(dim_proposals):
                # Append to the proposals a tuple (dimension, threshold) where the threshold is randomly chosen
                proposals.append((dim, np.random.choice(
                    np.unique(node_x[:, dim].flatten()), size=1, replace=False)[0]))

            # compute criterion for each proposal
            split_scores = np.zeros(len(proposals))
            for idx, (prop_feat, prop_thr) in enumerate(proposals):
                # if splitting creates valid leafs in terms of mean leaf size
                split_ratio_split = np.sum(
                    node_x[:, prop_feat] < prop_thr) / node_x.shape[0]
                min_leaf_size_split = min(
                    split_ratio_split * node_x.shape[0], (1 - split_ratio_split) * node_x.shape[0])
                split_ratio_estimate = np.sum(
                    node_x_estimate[:, prop_feat] < prop_thr) / node_x_estimate.shape[0]
                min_leaf_size_est = min(
                    split_ratio_estimate * node_x_estimate.shape[0], (1 - split_ratio_estimate) * node_x_estimate.shape[0])
                if min(min_leaf_size_split, min_leaf_size_est) > self.min_leaf_size and min(split_ratio_split, 1 - split_ratio_split) >= .5 - self.balancedness_tol and min(split_ratio_estimate, 1 - split_ratio_estimate) >= .5 - self.balancedness_tol:
                    # Calculate criterion for split
                    left_ind = node_x[:, prop_feat].flatten() < prop_thr
                    right_ind = node_x[:, prop_feat].flatten() >= prop_thr
                    rho_left = rho[left_ind]
                    rho_right = rho[right_ind]
                    left_score = - rho_left.mean() ** 2
                    right_score = - rho_right.mean() ** 2
                    split_scores[idx] = (right_score + left_score) / 2
                else:
                    # Else set criterion to infinity so that this split is not chosen
                    split_scores[idx] = np.inf

            # If no good split was found
            if np.min(split_scores) == np.inf:
                return node

            # Find split that minimizes criterion
            best_split_ind = np.argmin(split_scores)
            best_split_feat, best_split_thr = proposals[best_split_ind]

            # Set the split attributes at the node
            node.feature = best_split_feat
            node.threshold = best_split_thr

            # Create child nodes with corresponding subsamples
            left_split_sample_inds = node.split_sample_inds[node_x[:, best_split_feat].flatten(
            ) < best_split_thr]
            left_est_sample_inds = node.est_sample_inds[node_x_estimate[:, best_split_feat].flatten(
            ) < best_split_thr]
            node.left = Node(left_split_sample_inds, left_est_sample_inds)
            right_split_sample_inds = node.split_sample_inds[node_x[:, best_split_feat].flatten(
            ) >= best_split_thr]
            right_est_sample_inds = node.est_sample_inds[node_x_estimate[:, best_split_feat].flatten(
            ) >= best_split_thr]
            node.right = Node(right_split_sample_inds, right_est_sample_inds)

            # Recursively split children
            self.recursive_split(node.left, split_acc + 1)
            self.recursive_split(node.right, split_acc + 1)

            # Return parent node
            return node

    def create_splits(self):
        n = self.W.shape[0] // 2
        root = Node(np.arange(n), np.arange(
            n, self.W.shape[0]))
        self.tree = self.recursive_split(root, 0)

    def estimate_leafs(self, node):
        if node.left or node.right:
            self.estimate_leafs(node.left)
            self.estimate_leafs(node.right)
        else:
            # estimate the local parameter at the leaf using the estimate data
            node.estimate, _, _ = self.residualizer(self.W[node.est_sample_inds, :],
                                                    self.T[node.est_sample_inds],
                                                    self.Y[node.est_sample_inds],
                                                    model_T=self.model_T, model_Y=self.model_Y)
            node.est_sample_inds_1, node.est_sample_inds_2 = train_test_split(node.est_sample_inds, test_size=0.5)

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
