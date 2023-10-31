Federated Learning in EconML
============================
.. contents::
    :local:
    :depth: 2

Overview
--------

Federated Learning in the EconML Library allows models to be trained on separate data sets and then combined 
into a single CATE model afterwards, without ever needing to collect all of the training data on a single machine.

Motivation for Incorporating Federated Learning into the EconML Library
-----------------------------------------------------------------------

1. **Large data sets**: With data sets that are so large that they cannot fit onto a single machine, federated 
learning allows you to partition the data, train an individual causal model on each partition, and combine the models 
into a single model afterwards.  

2. **Privacy Preservation**: Federated learning enables organizations to build machine learning models without 
centralizing or sharing sensitive data.  This may be important to comply with data privacy regulations by keeping 
data localized and reducing exposure to compliance risks.

Federated Learning with EconML
------------------------------

Introducing the `FederatedEstimator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide the :class:`.FederatedEstimator` class to allow aggregating individual estimators which have 
been trained on different subsets of data.  The individual estimators must all be of the same type, 
which must currently be either LinearDML, LinearDRLearner, LinearDRIV, or LinearIntentToTreatDRIV.

Unlike other estimators, you should not call `fit` on an instance of :class:`.FederatedEstimator`; instead, 
you should train your individual estimators separately and then pass the already trained models to the :class:`.FederatedEstimator`
initializer.  The :class:`.FederatedEstimator` will then aggregate the individual estimators into a single model.


Example Usage
~~~~~~~~~~~~~

.. testsetup::

    import numpy as np
    from econml.federated_learning import FederatedEstimator
    from econml.dml import LinearDML
    n = 1000
    (X, y, t) = (np.random.normal(size=(n,)+s) for s in [(3,), (), ())

.. testcode::

    # Create individual LinearDML estimators
    num_partitions = 3
    estimators = []
    for i in range(num_partitions):
        est = LinearDML(random_state=123)
        # Get the data for this partition
        X_part, y_part, t_part = (arr[i::num_partitions] for arr in (X, y, t))

        # In practice, each estimator could be trained in a distributed fashion
        # e.g. by using Spark
        est.fit(Y=y_part, T=t_part, X=X_part)
        estimators.append(est)

    # Create a FederatedEstimator by providing a list of estimators
    federated_estimator = FederatedEstimator(estimators)

    # The federated estimator can now be used like a typical CATE estimator
    cme = federated_estimator.const_marginal_effect(X)



Theory
------

Many estimators are solving a moment equation

.. math::

    \E[\psi(D; \theta; \eta)] = 0

where :math:`D` is the data, :math:`\theta` is the parameter, and :math:`\eta` is the nuisance parameter.  Often, the moment is linear in the parameter, so that it can be rewritten as

.. math::

    \E[\psi_a(D; \eta)\theta + \psi_b(D; \eta)] = 0

In this case, solving the equation using the empirical expectations gives

.. math::

    \begin{align*}
        \hat{\theta} &= -\E_n[\psi_a(D;\hat{\eta})]^{-1} \E_n[\psi_b(D;\hat{\eta})] \\
        \sqrt{N}(\theta-\hat{\theta}) &\sim \mathcal{N}\left(0, \E_n[\psi_a(D;\hat{\eta})]^{-1} \E_n[\psi(D;\hat{\theta};\hat{\eta}) \psi(D;\hat{\theta};\hat{\eta})^\top] \E_n[\psi_a(D;\hat{\eta})^\top]^{-1}\right)
    \end{align*}

The center term in the variance calculation can be expanded out:

..  math::
    :nowrap:

    \begin{align*}
        \E_n[\psi(D;\hat\theta;\hat\eta) \psi(D;\hat\theta;\hat\eta)^\top] &= \E_n[(\psi_b(D;\hat\eta)+\psi_a(D;\hat\eta)\hat\theta) (\psi_b(D;\hat\eta)+\psi_a(D;\hat\eta)\hat\theta)^\top] \\
        &= \E_n[\psi_b(D;\hat\eta) \psi_b(D;\hat\eta)^\top] +  \E_n[\psi_a(D;\hat\eta)\hat\theta\psi_b(D;\hat\eta)^\top] \\ 
        &+ \E_n[\psi_b(D;\hat\eta) \hat\theta^\top \psi_a(D;\hat\eta)^\top] +  \E_n[\psi_a(D;\hat\eta) \hat\theta\hat\theta^\top\psi_a(D;\hat\eta)^\top ]
    \end{align*}

Some of these terms involve products where :math:`\hat\theta` appears in an interior position, but these can equivalently be computed by taking the outer product of the matrices on either side and then contracting with :math:`\hat\theta` afterwards.  Thus, we can distribute the computation of the following quantities:

.. math::
    :nowrap:

    \begin{align*}
        & \E_n[\psi_a(D;\hat\eta)] \\
        & \E_n[\psi_b(D;\hat\eta)] \\
        & \E_n[\psi_b(D;\hat\eta) \psi_b(D;\hat\eta)^\top] \\
        & \E_n[\psi_b(D;\hat\eta) \otimes \psi_a(D;\hat\eta)] \\
        & \E_n[\psi_a(D;\hat\eta) \otimes \psi_a(D;\hat\eta)] \\ 
    \end{align*}

We can then aggregate these distributed estimates, use the first two to calculate :math:`\hat\theta`, and then use that with the rest to calculate the analytical variance.

As an example, for linear regression of :math:`y` on :math:`X`, we have

.. math::

    \psi_a(D;\eta) = X^\top X \\
    \psi_b(D;\eta) = X^\top y

And so the additional moments we need to distribute are

.. math::

    \begin{align*}
        & \E_n[X^\top y y^\top X] = \E_n[X^\top X y^2] = \E_n[X \otimes X \otimes y \otimes y] \\
        & \E_n[X^\top y \otimes X^\top X] = \E_n[X \otimes X \otimes X \otimes y]\\
        & \E_n[X^\top X \otimes X^\top X] = \E_n[X \otimes X \otimes X \otimes X] \\ 
    \end{align*}

Thus, at the cost of storing these three extra moments, we can distribute the computation of linear regression and recover exactly the same 
result we would have gotten by doing this computation on the full data set.

In the context of federated CATE estimation, note that in practice the nuisances are computed on subsets of the data, 
so while it is true that the aggregated final linear model is exactly the same as what would be computed with all of the same nuisances locally,
in practice the nuisance estimates would differ if computed on all of the data.  In practice, this should not be a significant issue as long as the
nuisance estimators converge at a reasonable rate; for example if the first stage models are accurate enough for the final estimate to converge at a rate of :math:`O(1/\sqrt{n})`,
then splitting the data into :math:`k` partitions should only increase the variance by a factor of :math:`\sqrt{k}`.