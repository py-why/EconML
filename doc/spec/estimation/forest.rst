Forest Based Estimators
=======================

\

Orthogonal Random Forests
-------------------------

Orthogonal Random Forests [Oprescu2018]_ are a combination of causal forests and double machine learning that allow
for controlling for a high-dimensional set of confounders :math:`W`, while at the same time estimating non-parametrically
the heterogeneous treatment effect :math:`\theta(X)`, on a lower dimensional set of variables :math:`X`. 
Moreover, the estimates are asymptotically normal and hence have theoretical properties
that render bootstrap based confidence intervals asymptotically valid. 

In the case of continuous treatments (see :py:class:`~econml.ortho_forest.ContinuousTreatmentOrthoForest`) the method estimates :math:`\theta(x)` for some target :math:`x` by solving the following
system of equations:

.. math::

    \sum_{i=1}^n K(X_i, x)\cdot \left( Y_i - \hat{\E}[Y \mid x, W_i] - \langle \theta(x), T_i - \hat{\E}[T \mid x, W_i] \rangle \right)\cdot \left(T_i - \hat{\E}[T_i \mid x, W_i]\right) = 0

where :math:`\hat{\E}[Y \mid x, W_i]` and :math:`\hat{\E}[T \mid x, W_i]` are first stage estimates of the
corresponding conditional expectations. This approach is similar to the orthogonal/double machine learning
approach since we essentially perform a residual outcome on residual treatment regression. However, instead
of running an arbitrary regression we perform a non-parametric local weighted regression. The kernel :math:`K(X_i, x)`
is a similarity metric that is calculated by building a random forest with a causal criterion. This 
criterion is a slight modification of the criterion used in generalized random forests [Athey2019]_ and 
causal forests [Wager2018]_, so as to incorporated residualization when calculating the score of each candidate
split.

The method splits the data and performs cross-fitting: i.e. fit the
conditional expectation models on the first half and predicts the quantities on the second half and vice versa. 
Subsequently estimates :math:`\theta(x)` on all the data. 

In order to handle high-dimensional :math:`W`, the method estimates the conditional expectations also in a local manner
around each target :math:`x`. In particular, to estimate :math:`\hat{\E}[Y_i \mid x, W_i]` for each target :math:`x`
it minimizes a weighted (penalized) loss :math:`\ell` (e.g. square loss or multinomial logistic loss):

.. math::

    \min_{h_x \in H} \sum_{i=1}^n K(X_i, x)\cdot \ell(Y_i, h_x(W_i)) + R(h_x)

where :math:`H` is some function space and :math:`R` is some regularizer. If the hypothesis space
is locally linear, i.e. :math:`h_x(W) = \langle \nu(x), W \rangle`, the regularizer is the 
:math:`\ell_1` norm of the coefficients :math:`\|\nu(x)\|_1` and the loss is either the square
loss or the logistic loss, then the method has provable guarantees of asymptotic normality,
assuming the true coefficients are relatively sparse (i.e. most of them are zero). The 
weights :math:`K(X_i, x)` are computed using the same Random Forest algorithm with 
a causal criterion as the one used to calculate the weights for the second stage 
estimation of :math:`\theta(x)` (albeit using a different half sample than the one used for 
the final stage estimation, in a cross-fitting manner).

Algorithmically, the nuisance estimation part of the method is implemented in a
flexible manner, not restricted to :math:`\ell_1` regularization, as follows: the user can define any class that
supports fit and predict. The fit function needs to also support sample weights, passed as a third argument. 
If it does not, then we provided a weighted model wrapper :py:class:`~econml.ortho_forest.WeightedModelWrapper` that
can wrap any class that supports fit and predict and enables sample weight functionality. This is done either
by re-sampling the data based on the weights and then calling fit and predict, or, in the case of square losses of
linear function classes, by re-scaling the features and labels appropriately based on the weights.

    >>> est = ContinuousTreatmentOrthoForest(model_y=WeightedModelWrapper(Lasso(), sample_type=sample_type),
    ...                                      model_t=WeightedModelWrapper(Lasso(), sample_type=sample_type))

If the variable :code:`sample_type` takes the value "weighted", then the wrapper assumes the loss
is the squared loss and the function class is linear and re-scales the features and labels appropriately.
If not, then it re-samples the data based on the weights and calls the fit method of the base
class on this re-sampled dataset. The latter has higher variance and should not be chosen if the
first approach is applicable.

In the case of discrete treatments (see :py:class:`~econml.ortho_forest.DiscreteTreatmentOrthoForest`) the
method estimates :math:`\theta(x)` for some target :math:`x` by solving a slightly different
set of equations (see [Oprescu2018]_ for a theoretical exposition of why a different set of
estimating equations is used). In particular, suppose that the treatment :math:`T` takes
values in :math:`\{0, 1, \ldots, k\}`, then to estimate the treatment effect :math:`\theta_t(x)` of
treatment :math:`t` as compared to treatment :math:`0`, the method finds the solution to the
equation:

.. math::

    \sum_{i=1}^n K(X_i, x)\cdot \left( Y_{i,t}^{DR} - Y_{i,0}^{DR}- \theta_t(x) \right) = 0

where :math:`Y_{i,t}^{DR}` is a doubly robust based unbiased estimate of the counterfactual
outcome of sample :math:`i` had we treated it with treatment :math:`t`, i.e.:

.. math::
    
    Y_{i,t}^{DR} = \hat{\E}[Y \mid T=t, x, W_i] + 1\{T_i=t\} \frac{Y_i - \hat{\E}[Y \mid T=t, x, W_i]}{\hat{\E}[1\{T=t\} \mid x, W_i]} 

where :math:`\hat{\E}[Y \mid T=t, x, W_i]` and :math:`\hat{\E}[1\{T=t\} \mid x, W_i]` are first stage estimates of the
corresponding conditional expectations. These two regression functions are fitted in a similar manner
as in the continuous treatment case. However, in the case of discrete treatment, the model for the treatment is 
a multi-class classification model and should support :code:`predict_proba`.    

For more details on the input parameters of the orthogonal forest classes and how to customize
the estimator checkout the two modules:

- :py:class:`~econml.ortho_forest.DiscreteTreatmentOrthoForest`
- :py:class:`~econml.ortho_forest.ContinuousTreatmentOrthoForest`

For more examples check out our 
`OrthoForest Jupyter notebook <https://github.com/Microsoft/EconML/blob/master/notebooks/Orthogonal%20Random%20Forest%20Examples.ipynb>`_ 

Examples
^^^^^^^^

Here is a simple example of how to call :py:class:`~econml.ortho_forest.ContinuousTreatmentOrthoForest`
and what the returned values correspond to in a simple data generating process:

    >>> T = np.array([0, 1]*60)
    >>> W = np.array([0, 1, 1, 0]*30).reshape(-1, 1)
    >>> Y = (.2 * W[:, 0] + 1) * T + .5
    >>> est = ContinuousTreatmentOrthoForest(n_trees=1, max_splits=1, subsample_ratio=1,
    ...                                      model_T=sklearn.linear_model.LinearRegression(),
    ...                                      model_Y=sklearn.linear_model.LinearRegression())
    >>> est.fit(Y, T, W, W)
    >>> print(est.const_marginal_effect(W[:2]))
    [[1. ]
     [1.2]]

Similarly, we can call :py:class:`~econml.ortho_forest.DiscreteTreatmentOrthoForest`:

    >>> T = np.array([0, 1]*60)
    >>> W = np.array([0, 1, 1, 0]*30).reshape(-1, 1)
    >>> Y = (.2 * W[:, 0] + 1) * T + .5
    >>> est = DiscreteTreatmentOrthoForest(n_trees=1, max_splits=1, subsample_ratio=1,
    ...                                    propensity_model=sklearn.linear_model.LogisticRegression(),
    ...                                    model_Y=sklearn.linear_model.LinearRegression())
    >>> est.fit(Y, T, W, W)
    >>> print(est.const_marginal_effect(W[:2]))
    [[1. ]
     [1.2]]

Let's now look at a more involved example with a high-dimensional set of confounders :math:`W`
and with more realistic noisy data. In this case we can just use the default Parameters
of the class, which specify the use of the :py:class:`~sklearn.linear_model.LassoCV` for 
both the treatment and the outcome regressions, in the case of continuous treatments.

    >>> X = np.random.uniform(-1, 1, size=(4000, 1))
    >>> W = np.random.normal(size=(4000, 50))
    >>> support = np.random.choice(50, 4, replace=False)
    >>> T = np.dot(W[:, support], np.random.normal(size=4)) + np.random.normal(size=4000)
    >>> Y = np.exp(2*X[:, 0]) * T + np.dot(W[:, support], np.random.normal(size=4)) + .5
    >>> est = ContinuousTreatmentOrthoForest()
    >>> est.fit(Y, T, X, W)
    >>> X_test = np.linspace(-1, 1, 30).reshape(-1, 1)
    >>> treatment_effects = est.const_marginal_effect(X_test)
    >>> plt.plot(X_test, y, label='ORF estimate')
    >>> plt.plot(X_test[:, 0], np.exp(2*X_test[:, 0]), 'b--', label='True effect')
    >>> plt.legend()
    >>> plt.show()

.. figure:: figures/continuous_ortho_forest_doc_example.png
    :align: center

    Synthetic data estimation with high dimensional controls
