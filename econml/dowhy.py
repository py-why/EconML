# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


"""Helper class to allow other functionalities from dowhy package.

References
----------
DoWhy, https://microsoft.github.io/dowhy/

"""

import inspect
import pandas as pd
import numpy as np
from dowhy import CausalModel
from econml.utilities import check_input_arrays, reshape_arrays_2dim


class DoWhyWrapper:
    """A wrapper class to allow user call other methods from dowhy package through EconML.
    (e.g. causal graph, refutation test, etc.)

    Parameters
    ----------
    cate_estimator: instance
        An instance of any CATE estimator we currently support

    """

    def __init__(self, cate_estimator):
        self._cate_estimator = cate_estimator

    def _get_params(self):
        init = self._cate_estimator.__init__
        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("cate estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (self._cate_estimator, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def fit(self, Y, T, X=None, W=None, Z=None, *, outcome_names=None, treatment_names=None, feature_names=None,
            confounder_names=None, instrument_names=None, graph=None, estimand_type="nonparametric-ate",
            proceed_when_unidentifiable=True, missing_nodes_as_confounders=False,
            control_value=0, treatment_value=1, target_units="ate", **kwargs):
        """Estimate the counterfactual model from data through dowhy package.

        Parameters
        ----------
        Y: vector of length n
            Outcomes for each sample
        T: vector of length n
            Treatments for each sample
        X: optional (n, d_x) matrix (Default=None)
            Features for each sample
        W: optional (n, d_w) matrix (Default=None)
            Controls for each sample
        Z: optional (n, d_z) matrix (Default=None)
            Instruments for each sample
        outcome_names: optional list (Default=None)
            Name of the outcome
        treatment_names: optional list (Default=None)
            Name of the treatment
        feature_names: optional list (Default=None)
            Name of the features
        confounder_names: optional list (Default=None)
            Name of the confounders
        instrument_names: optional list (Default=None)
            Name of the instruments
        graph: optional
            Path to DOT file containing a DAG or a string containing a DAG specification in DOT format
        estimand_type: optional string
            Type of estimand requested (currently only "nonparametric-ate" is supported).
            In the future, may support other specific parametric forms of identification
        proceed_when_unidentifiable: optional bool (Default=True)
            Whether the identification should proceed by ignoring potential unobserved confounders
        missing_nodes_as_confounders: optional bool (Default=False)
            Whether variables in the dataframe that are not included in the causal graph should be automatically
            included as confounder nodes
        control_value: optional scalar (Default=0)
            Value of the treatment in the control group, for effect estimation
        treatment_value: optional scalar (Default=1)
            Value of the treatment in the treated group, for effect estimation
        target_units: optional (Default="ate")
            The units for which the treatment effect should be estimated.
            This can be of three types:
                (1) a string for common specifications of target units (namely, "ate", "att" and "atc"),
                (2) a lambda function that can be used as an index for the data (pandas DataFrame),
                (3) a new DataFrame that contains values of the effect_modifiers and effect will be estimated
                 only for this new data
        kwargs: optional
            Other keyword arguments from fit method for CATE estimator

        Returns
        -------
        self

        """

        Y, T, X, W, Z = check_input_arrays(Y, T, X, W, Z)

        # create dataframe
        n_obs = Y.shape[0]
        Y, T, X, W, Z = reshape_arrays_2dim(n_obs, Y, T, X, W, Z)

        # currently dowhy only support single outcome and single treatment
        assert Y.shape[1] == 1, "Can only accept single dimensional outcome."
        assert T.shape[1] == 1, "Can only accept single dimensional treatment."

        # column names
        if outcome_names is None:
            outcome_names = [f"Y{i}" for i in range(Y.shape[1])]
        if treatment_names is None:
            treatment_names = [f"T{i}" for i in range(T.shape[1])]
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X.shape[1])]
        if confounder_names is None:
            confounder_names = [f"W{i}" for i in range(W.shape[1])]
        if instrument_names is None:
            instrument_names = [f"Z{i}" for i in range(Z.shape[1])]
        column_names = outcome_names + treatment_names + feature_names + confounder_names + instrument_names
        df = pd.DataFrame(np.hstack((Y, T, X, W, Z)), columns=column_names)
        self.dowhy_ = CausalModel(
            data=df,
            treatment=treatment_names,
            outcome=outcome_names,
            graph=graph,
            common_causes=feature_names + confounder_names if X.shape[1] > 0 or W.shape[1] > 0 else None,
            instruments=instrument_names if Z.shape[1] > 0 else None,
            effect_modifiers=feature_names if X.shape[1] > 0 else None,
            estimand_type=estimand_type,
            proceed_when_unidetifiable=proceed_when_unidentifiable,
            missing_nodes_as_confounders=missing_nodes_as_confounders
        )
        self.identified_estimand_ = self.dowhy_.identify_effect(proceed_when_unidentifiable=True)
        method_name = "backdoor." + self._cate_estimator.__module__ + "." + self._cate_estimator.__class__.__name__
        init_params = {}
        for p in self._get_params():
            init_params[p] = getattr(self._cate_estimator, p)
        self.estimate_ = self.dowhy_.estimate_effect(self.identified_estimand_,
                                                     method_name=method_name,
                                                     control_value=control_value,
                                                     treatment_value=treatment_value,
                                                     target_units=target_units,
                                                     method_params={
                                                         "init_params": init_params,
                                                         "fit_params": kwargs,
                                                     },
                                                     )
        return self

    def refute_estimate(self, *, method_name, **kwargs):
        """Refute an estimated causal effect.

        If method_name is provided, uses the provided method. In the future, we may support automatic
        selection of suitable refutation tests.
        Following refutation methods are supported:
            * Adding a randomly-generated confounder: "random_common_cause"
            * Adding a confounder that is associated with both treatment and outcome: "add_unobserved_common_cause"
            * Replacing the treatment with a placebo (random) variable): "placebo_treatment_refuter"
            * Removing a random subset of the data: "data_subset_refuter"

        Parameters
        ----------
        method_name: string
            Name of the refutation method
        kwargs: optional
            Additional arguments that are passed directly to the refutation method.
            Can specify a random seed here to ensure reproducible results ('random_seed' parameter).
            For method-specific parameters, consult the documentation for the specific method.
            All refutation methods are in the causal_refuters subpackage.

        Returns
        -------
        RefuteResult: an instance of the RefuteResult class
        """
        return self.dowhy_.refute_estimate(
            self.identified_estimand_, self.estimate_, method_name=method_name, **kwargs
        )

    # We don't allow user to call refit_final from this class, since internally dowhy effect estimate will only update
    # cate estimator but not the effect.
    def refit_final(self, inference=None):
        raise AttributeError(
            "Method refit_final is not allowed here! Please call it from Cate Estimator directly! ")

    def __getattr__(self, attr):
        # don't proxy special methods
        if attr.startswith('__'):
            raise AttributeError(attr)
        elif attr in ['_cate_estimator', 'dowhy_',
                      'identified_estimand_', 'estimate_']:
            self.__dict__[attr] = value
        elif attr.startswith('dowhy__'):
            return getattr(self.dowhy_, attr[7:])
        elif hasattr(self.estimate_._estimator_object, attr):

            return getattr(self.estimate_._estimator_object, attr)
        else:
            return getattr(self.dowhy_, attr)

    def __setattr__(self, attr, value):
        if attr in ['_cate_estimator', 'dowhy_',
                    'identified_estimand_', 'estimate_']:
            self.__dict__[attr] = value
        elif attr.startswith('dowhy__'):
            setattr(self.dowhy_, attr[7:], value)
        elif hasattr(self.estimate_._estimator_object, attr):
            setattr(self.estimate_._estimator_object, attr, value)
        else:
            setattr(self.dowhy_, attr, value)
