Instrumental Variable Regression
================================

Instrumental variables (IV) methods are an approach for estimating causal effects despite the presence of confounding latent variables.  
The assumptions made are weaker than the unconfoundedness assumption needed in DML.
The cost is that when unconfoundedness holds, IV estimators will be less efficient than DML estimators.  
What is required is a vector of instruments :math:`Z`, assumed to casually affect the distribution of the treatment :math:`T`, 
and to have no direct causal effect on the expected value of the outcome :math:`Y`.  The package offers two IV methods for 
estimating heterogeneous treatment effects: deep instrumental variables [Hartford2017]_ and the two-stage basis expansion approach 
of [Newey2003]_.  

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

Deep Instrumental Variables
---------------------------

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

Sieve Instrumental Variable Estimation
--------------------------------------

The sieve based instrumental variable module is based on a two-stage least squares estimation procedure.
The user must specify the sieve basis for :math:`T`, :math:`X` and :math:`Y` (Hermite polynomial or a set of indicator 
functions), and the number of elements of the basis expansion to include. Formally, we now assume that we can write:

.. math::

    Y =~& \sum_{d=1}^{d^Y} \sum_{k=1}^{d^X} \beta^Y_{d,k} \psi_d(T) \rho_k(X) + \gamma (X,W) + \epsilon \\
    T =~& \sum_{d=1}^{d^T} \sum_{k=1}^{d^X} \beta^T_{d,k} \phi_d(Z) \rho_k(X) + \delta (X,W) + u

where :math:`\{\psi_d\}` is the sieve basis for :math:`Y` with degree :math:`d^Y`, :math:`\{\rho_k\}` is the sieve basis 
for :math:`X`, with degree :math:`d^X`, :math:`\{\phi_d\}` is the sieve basis for :math:`T` with degree :math:`d^T`, 
:math:`Z` are the instruments, :math:`(X,W)` is the horizontal concatenation of :math:`X` and :math:`W`, and :math:`u` 
and :math:`\varepsilon` may be correlated. Each of the :math:`\psi_d` is a function from :math:`\dim(T)` into 
:math:`\mathbb{R}`, each of the :math:`\rho_k` is a function from :math:`\dim(X)` into :math:`\mathbb{R}` and each 
of the :math:`\phi_d` is a function from :math:`\dim(Z)` into :math:`\mathbb{R}`.  

Our goal is to estimate

.. math::

    \tau(\vec{t}_0, \vec{t}_1, \vec{x}) = \sum_{d=1}^{d^Y} \sum_{k=1}^{d^X} \beta^Y_{d,k} \rho_k(\vec{x})  \left(\psi_d(\vec{t_1}) - \psi_d(\vec{t_0})\right)

We do this by first estimating each of the functions :math:`\E[\psi_d(T)|X,Z,W]` by linear projection of :math:`\psi_d(t_i)` 
onto the features :math:`\{\phi_d(z_i) \rho_k(x_i) \}` and :math:`(x_i,w_i)`. We will then project :math:`y_i` onto these
estimated functions and :math:`(x_i,w_i)` again to arrive at an estimate :math:`\hat{\beta}^Y` whose individual coefficients 
:math:`\beta^Y_{d,k}` can be used to return our estimate of :math:`\tau`.  

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

