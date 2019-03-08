Forest Based Estimators
=======================


Orthogonal Random Forests
-------------------------

Orthogonal Random Forests are a combination of causal forests and double machine learning that allow
for controlling for a high-dimensional set of confounders :math:`W`, while at the same time estimating non-parametrically
the heterogeneous treatment effect :math:`\theta(X)`, on a lower dimensional set of variables :math:`X`. 
Moreover, the estimates are asymptotically normal and hence have theoretical properties
that render bootstrap based confidence intervals asymptotically valid. 

In the case of continuous treatments (see :py:class:`~econml.ortho_forest.ContinuousTreatmentOrthoForest`) the method estimates :math:`\theta(x)` for some target :math:`x` by solving the following
system of equations:

.. math::

    \sum_{i=1}^n K(X_i, x)\cdot \left( Y_i - \hat{\E}[Y_i \mid x, W_i] - \langle \theta(x), T_i - \hat{\E}[T_i \mid x, W_i] \rangle \right)\cdot \left(T_i - \hat{\E}[T_i \mid x, W_i]\right) = 0

where :math:`\hat{\E}[Y_i \mid x, W_i]` and :math:`\hat{\E}[T_i \mid x, W_i]` are first stage estimates of the
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
assuming the true coefficients are relatively sparse (i.e. most of them are zero). 

Algorithmically, the nuisance estimation part of the method is implemented as follows: the user can define any class that
supports fit and predict. The fit function needs to also support sample weights, passed as a third argument. 
If it does not, then we provided a weighted model wrapper :py:class:`~econml.ortho_forest.WeightedModelWrapper` that
can wrap any class that supports fit and predict and enables sample weight functionality. This is done either
by re-sampling the data based on the weights and then calling fit and predict, or, in the case of linear losses,
by re-scaling the features and labels appropriately based on the weights.


    >>> est = ContinuousTreatmentOrthoForest(model_y=WeightedModelWrapper(Lasso(), sample_type=sample_type),
    ...                                      model_t=WeightedModelWrapper(Lasso(), sample_type=sample_type))

If the variable :code:`sample_type` takes the value "weighted", then the wrapper assumes the loss
is the squared loss and the function class is linear and re-scales the features and labels appropriately.
If not, then it re-samples the data based on the weights and calls the fit method of the base
class on this re-sampled dataset. The latter has higher variance and should not be chosen if the
first approach is applicable.
