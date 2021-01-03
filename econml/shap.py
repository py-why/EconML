# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import shap
from collections import defaultdict
import numpy as np


def _shap_explain_cme(cme_model, X, d_t, d_y, feature_names=None, treatment_names=None, output_names=None):
    """
    Method to explain `const_marginal_effect` function using shap Explainer().

    Parameters
    ----------
    cme_models: function
        const_marginal_effect function.
    X: (m, d_x) matrix
        Features for each sample. Should be in the same shape of fitted X in final stage.
    d_t: tuple of int
        Tuple of number of treatment (exclude control in discrete treatment scenario).
    d_y: tuple of int
        Tuple of number of outcome.
    feature_names: optional None or list of strings of length X.shape[1] (Default=None)
        The names of input features.
    treatment_names: optional None or list (Default=None)
        The name of treatment. In discrete treatment scenario, the name should not include the name of
        the baseline treatment (i.e. the control treatment, which by default is the alphabetically smaller)
    output_names:  optional None or list (Default=None)
        The name of the outcome.

    Returns
    -------
    shap_outs: nested dictionary of Explanation object
        A nested dictionary by using each output name (e.g. "Y0" when `output_names=None`) and
        each treatment name (e.g. "T0" when `treatment_names=None`) as key
        and the shap_values explanation object as value.

    """
    (dt, dy, treatment_names, output_names) = _define_names(d_t, d_y, treatment_names, output_names)
    # define masker by using entire dataset, otherwise Explainer will only sample 100 obs by default.
    background = shap.maskers.Independent(X, max_samples=X.shape[0])
    shap_outs = defaultdict(dict)
    for i in range(dy):
        def cmd_func(X):
            return cme_model(X).reshape(-1, dy, dt)[:, i, :]
        explainer = shap.Explainer(cmd_func, background,
                                   feature_names=feature_names)
        shap_out = explainer(X)
        if dt > 1:
            for j in range(dt):
                base_values = shap_out.base_values[..., j]
                values = shap_out.values[..., j]
                main_effects = None if shap_out.main_effects is None else shap_out.main_effects[..., j]
                shap_out_new = shap.Explanation(values, base_values=base_values,
                                                data=shap_out.data, main_effects=main_effects,
                                                feature_names=shap_out.feature_names)
                shap_outs[output_names[i]][treatment_names[j]] = shap_out_new
        else:
            base_values = shap_out.base_values[..., 0]
            shap_out_new = shap.Explanation(shap_out.values, base_values=base_values,
                                            data=shap_out.data, main_effects=shap_out.main_effects,
                                            feature_names=shap_out.feature_names)
            shap_outs[output_names[i]][treatment_names[0]] = shap_out_new
    return shap_outs


def _shap_explain_model_cate(cme_model, models, X, d_t, d_y, feature_names=None,
                             treatment_names=None, output_names=None):
    """
    Method to explain `model_cate` using shap Explainer(), will instead explain `const_marignal_effect`
    if `model_cate` can't be parsed.

    Parameters
    ----------
    cme_models: function
        const_marginal_effect function.
    models: a single estimator or a list of estimators with one estimator per treatment
        models for the model's final stage model.
    X: (m, d_x) matrix
        Features for each sample. Should be in the same shape of fitted X in final stage.
    d_t: tuple of int
        Tuple of number of treatment (exclude control in discrete treatment scenario.
    d_y: tuple of int
        Tuple of number of outcome.
    feature_names: optional None or list of strings of length X.shape[1] (Default=None)
        The names of input features.
    treatment_names: optional None or list (Default=None)
        The name of treatment. In discrete treatment scenario, the name should not include the name of
        the baseline treatment (i.e. the control treatment, which by default is the alphabetically smaller)
    output_names:  optional None or list (Default=None)
        The name of the outcome.

    Returns
    -------
    shap_outs: nested dictionary of Explanation object
        A nested dictionary by using each output name (e.g. "Y0" when `output_names=None`) and
        each treatment name (e.g. "T0" when `treatment_names=None`) as key
        and the shap_values explanation object as value.
    """

    (dt, dy, treatment_names, output_names) = _define_names(d_t, d_y, treatment_names, output_names)
    if not isinstance(models, list):
        models = [models]
    assert len(models) == dt, "Number of final stage models don't equals to number of treatments!"
    # define masker by using entire dataset, otherwise Explainer will only sample 100 obs by default.
    background = shap.maskers.Independent(X, max_samples=X.shape[0])

    shap_outs = defaultdict(dict)
    for i in range(dt):
        try:
            explainer = shap.Explainer(models[i], background,
                                       feature_names=feature_names)
        except Exception as e:
            print("Final model can't be parsed, explain const_marginal_effect() instead!")
            return _shap_explain_cme(cme_model, X, d_t, d_y, feature_names, treatment_names,
                                     output_names)
        shap_out = explainer(X)
        if dy > 1:
            for j in range(dy):
                base_values = shap_out.base_values[..., j]
                values = shap_out.values[..., j]
                main_effects = None if shap_out.main_effects is None else shap_out.main_effects[..., j]
                shap_out_new = shap.Explanation(values, base_values=base_values,
                                                data=shap_out.data, main_effects=main_effects,
                                                feature_names=shap_out.feature_names)
                shap_outs[output_names[j]][treatment_names[i]] = shap_out_new
        else:
            shap_outs[output_names[0]][treatment_names[i]] = shap_out

    return shap_outs


def _shap_explain_joint_linear_model_cate(model_final, X, T, d_t, d_y, fit_cate_intercept,
                                          feature_names=None, treatment_names=None, output_names=None):
    """
    Method to explain `model_cate` of parametric final stage that was fitted on the cross product of
    `featurizer(X)` and T.

    Parameters
    ----------
    model_final: a single estimator
        the model's final stage model.
    X: matrix
        Intermediate X
    T: matrix
        Intermediate T
    d_t: tuple of int
        Tuple of number of treatment (exclude control in discrete treatment scenario).
    d_y: tuple of int
        Tuple of number of outcome.
    fit_cate_intercept: bool
        Whether the intercept is the first entry of the coefficient
    feature_names: optional None or list of strings of length X.shape[1] (Default=None)
        The names of input features.
    treatment_names: optional None or list (Default=None)
        The name of treatment. In discrete treatment scenario, the name should not include the name of
        the baseline treatment (i.e. the control treatment, which by default is the alphabetically smaller)
    output_names:  optional None or list (Default=None)
        The name of the outcome.

    Returns
    -------
    shap_outs: nested dictionary of Explanation object
        A nested dictionary by using each output name (e.g. "Y0" when `output_names=None`) and
        each treatment name (e.g. "T0" when `treatment_names=None`) as key
        and the shap_values explanation object as value.
    """

    d_x = X.shape[1]
    # define the index of d_x to filter for each given T
    ind_x = np.arange(d_x).reshape(d_t, -1)
    if fit_cate_intercept:  # skip intercept
        ind_x = ind_x[:, 1:]
    shap_outs = defaultdict(dict)
    for i in range(d_t):
        # filter X after broadcast with T for each given T
        X_sub = X[T[:, i] == 1]
        # define masker by using entire dataset, otherwise Explainer will only sample 100 obs by default.
        background = shap.maskers.Independent(X_sub, max_samples=X_sub.shape[0])
        explainer = shap.Explainer(model_final, background)
        shap_out = explainer(X_sub)

        data = shap_out.data[:, ind_x[i]]
        if d_y > 1:
            for j in range(d_y):
                base_values = shap_out.base_values[..., j]
                main_effects = shap_out.main_effects[..., ind_x[i], j]
                values = shap_out.values[..., ind_x[i], j]
                shap_out_new = shap.Explanation(values, base_values=base_values, data=data, main_effects=main_effects,
                                                feature_names=feature_names)
                shap_outs[output_names[j]][treatment_names[i]] = shap_out_new
        else:
            values = shap_out.values[..., ind_x[i]]
            main_effects = shap_out.main_effects[..., ind_x[i], 0]
            shap_out_new = shap.Explanation(values, base_values=shap_out.base_values, data=data,
                                            main_effects=main_effects,
                                            feature_names=feature_names)
            shap_outs[output_names[0]][treatment_names[i]] = shap_out_new

    return shap_outs


def _shap_explain_multitask_model_cate(cme_model, multitask_model_cate, X, d_t, d_y, feature_names=None,
                                       treatment_names=None, output_names=None):
    """
    Method to explain `multitask_model_cate` for DRLearner

    Parameters
    ----------
    cme_model: function
        const_marginal_effect function.
    multitask_model_cate: a single estimator
        the model's final stage model.
    X: (m, d_x) matrix
        Features for each sample. Should be in the same shape of fitted X in final stage.
    d_t: tuple of int
        Tuple of number of treatment (exclude control in discrete treatment scenario).
    d_y: tuple of int
        Tuple of number of outcome.
    feature_names: optional None or list of strings of length X.shape[1] (Default=None)
        The names of input features.
    treatment_names: optional None or list (Default=None)
        The name of treatment. In discrete treatment scenario, the name should not include the name of
        the baseline treatment (i.e. the control treatment, which by default is the alphabetically smaller)
    output_names:  optional None or list (Default=None)
        The name of the outcome.

    Returns
    -------
    shap_outs: nested dictionary of Explanation object
        A nested dictionary by using each output name (e.g. "Y0" when `output_names=None`) and
        each treatment name (e.g. "T0" when `treatment_names=None`) as key
        and the shap_values explanation object as value.
    """
    (dt, dy, treatment_names, output_names) = _define_names(d_t, d_y, treatment_names, output_names)

    # define masker by using entire dataset, otherwise Explainer will only sample 100 obs by default.
    background = shap.maskers.Independent(X, max_samples=X.shape[0])
    shap_outs = defaultdict(dict)
    try:
        explainer = shap.Explainer(multitask_model_cate, background,
                                   feature_names=feature_names)
    except Exception as e:
        print("Final model can't be parsed, explain const_marginal_effect() instead!")
        return _shap_explain_cme(cme_model, X, d_t, d_y, feature_names, treatment_names,
                                 output_names)

    shap_out = explainer(X)
    if dt > 1:
        for i in range(dt):
            base_values = shap_out.base_values[..., i]
            values = shap_out.values[..., i]
            main_effects = None if shap_out.main_effects is not None else shap_out.main_effects[..., i]
            shap_out_new = shap.Explanation(values, base_values=base_values,
                                            data=shap_out.data, main_effects=main_effects,
                                            feature_names=shap_out.feature_names)
            shap_outs[output_names[0]][treatment_names[i]] = shap_out_new
    else:
        shap_outs[output_names[0]][treatment_names[0]] = shap_out
    return shap_outs


def _define_names(d_t, d_y, treatment_names, output_names):
    """
    Helper function to get treatment and output names

    Parameters
    ----------
    d_t: tuple of int
        Tuple of number of treatment (exclude control in discrete treatment scenario).
    d_y: tuple of int
        Tuple of number of outcome.
    treatment_names: optional None or list (Default=None)
        The name of treatment. In discrete treatment scenario, the name should not include the name of
        the baseline treatment (i.e. the control treatment, which by default is the alphabetically smaller)
    output_names:  optional None or list (Default=None)
        The name of the outcome.

    Returns
    -------
    d_t: int
    d_y: int
    treament_names: List
    output_names: List
    """

    d_t = d_t[0] if d_t else 1
    d_y = d_y[0] if d_y else 1
    if treatment_names is None:
        treatment_names = [f"T{i}" for i in range(d_t)]
    if output_names is None:
        output_names = [f"Y{i}" for i in range(d_y)]
    return (d_t, d_y, treatment_names, output_names)
