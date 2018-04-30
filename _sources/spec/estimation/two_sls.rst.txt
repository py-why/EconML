
Sieve 2SLS Instrumental Variable Estimation
===========================================

The sieve based instrumental variable estimator :class:`.SieveTSLS` is based on a two-stage least squares estimation procedure.
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
