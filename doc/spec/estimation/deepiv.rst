Deep Instrumental Variables
===========================

Instrumental variables (IV) methods are an approach for estimating causal effects despite the presence of confounding latent variables.  
The assumptions made are weaker than the unconfoundedness assumption needed in DML.
The cost is that when unconfoundedness holds, IV estimators will be less efficient than DML estimators.  
What is required is a vector of instruments :math:`Z`, assumed to casually affect the distribution of the treatment :math:`T`, 
and to have no direct causal effect on the expected value of the outcome :math:`Y`.  The package offers two IV methods for
estimating heterogeneous treatment effects: deep instrumental variables [Hartford2017]_
and the two-stage basis expansion approach of [Newey2003]_.

The setup of the model is as follows:

.. math::

    Y = g(T, X, W) + \epsilon

where :math:`\E[\varepsilon|X,W,Z] = h(X,W)`, so that the expected value of :math:`Y` depends only on :math:`(T,X,W)`. 
This is known as the *exclusion restriction*.
We assume that the conditional distribution :math:`F(T|X,W,Z)` varies with :math:`Z`.
This is known as the *relevance condition*.
We want to learn the heterogeneous treatment effects: 

.. math::

    \tau(\vec{t}_0, \vec{t}_1, \vec{x}) = \E[g(\vec{t}_1,\vec{x},W) - g(\vec{t}_0,\vec{x},W)] 

where the expectation is taken with respect to the conditional distribution of :math:`W|\vec{x}`.
If the function :math:`g` is truly non-parametric, then in the special case where :math:`T`, :math:`Z` and :math:`X` are discrete, 
the probability matrix giving the distribution of :math:`T` for each value of :math:`Z` needs to be invertible pointwise at :math:`\vec{x}` 
in order for this quantity to be identified for arbitrary :math:`\vec{t}_0` and :math:`\vec{t}_1`.
In practice though we will place some parametric structure on the function :math:`g` which will make learning easier.
In deep IV, this takes the form of assuming :math:`g` is a neural net with a given architecture; in the sieve based approaches, 
this amounts to assuming that :math:`g` is a weighted sum of a fixed set of basis functions. [1]_

As explained in [Hartford2017]_, the Deep IV module learns the heterogenous causal effects by minimizing the "reduced-form" prediction error:

.. math::

    \hat{g}(T,X,W) \equiv \argmin_{g \in \mathcal{G}} \sum_i \left(y_i - \int g(T,x_i,w_i) dF(T|x_i,w_i,z_i)\right)^2 

where the hypothesis class :math:`\mathcal{G}` are neural nets with a given architecture.
The distribution :math:`F(T|x_i,w_i,z_i)` is unknown and so to make the objective feasible it must be replaced by an estimate 
:math:`\hat{F}(T|x_i,w_i,z_i)`.
This estimate is obtained by modeling :math:`F` as a mixture of normal distributions, where the parameters of the mixture model are 
the output of a "first-stage" neural net whose inputs are :math:`(x_i,w_i,z_i)`.  
Optimization of the "first-stage" neural net is done by stochastic gradient descent on the (mixture-of-normals) likelihood, 
while optimization of the "second-stage" model for the treatment effects is done by stochastic gradient descent with 
three different options for the loss:

    *   Estimating the two integrals that make up the true gradient calculation by independent averages over 
        mini-batches of data, which are unbiased estimates of the integral.
    *   Using the modified objective function 
    
        .. math::
        
            \sum_i \sum_d \left(y_i - g(t_d,x_i,w_i)\right)^2

        where :math:`t_d \sim \hat{F}(t|x_i,w_i,z_i)` are draws from the estimated first-stage neural net. This modified 
        objective function is not guaranteed to lead to consistent estimates of :math:`g`, but has the advantage of requiring
        only a single set of samples from the distribution, and can be interpreted as regularizing the loss with a 
        variance penalty. [2]_
    *   Using a single set of samples to compute the gradient of the loss; this will only be an unbiased estimate of the 
        gradient in the limit as the number of samples goes to infinity.

Training proceeds by splitting the data into a training and test set, and training is stopped when test set performance 
(on the reduced form prediction error) starts to degrade.  

The output is an estimated function :math:`\hat{g}`.  To obtain an estimate of :math:`\tau`, we difference the estimated 
function at :math:`\vec{t}_1` and :math:`\vec{t}_0`, replacing the expectation with the empirical average over all
observations with the specified :math:`\vec{x}`.    


.. rubric:: Footnotes

.. [1]
    Asymptotic arguments about non-parametric consistency require that the neural net architecture (respectively set of basis functions) 
    are allowed to grow at some rate so that arbitrary functions can be approximated, but this will not be our concern here.
.. [2]
    .. math::

        & \int \left(y_i - g(t,x_i,w_i)\right)^2 dt \\
        =~& y_i - 2 y_i \int g(t,x_i,w_i)\,dt + \int g(t,x_i,w_i)^2\,dt \\
        =~& y_i - 2 y_i \int g(t,x_i,w_i)\,dt + \left(\int g(t,x_i,w_i)\,dt\right)^2 + \int g(t,x_i,w_i)^2\,dt - \left(\int g(t,x_i,w_i)\,dt\right)^2 \\
        =~& \left(y_i - \int g(t,x_i,w_i)\,dt\right)^2 + \left(\int g(t,x_i,w_i)^2\,dt - \left(\int g(t,x_i,w_i)\,dt\right)^2\right) \\
        =~& \left(y_i - \int g(t,x_i,w_i)\,dt\right)^2 + \Var_t g(t,x_i,w_i)
