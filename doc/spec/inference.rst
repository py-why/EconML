Generic Inference
=================

\ 

Bootstrap Subsampling
---------------------

We provide a generic bootstrap sampling estimator :class:`.BootstrapEstimator` that can wrap either sklearn 
or econml estimators.  This requires the wrapped object to provide a fit method, whose signature will be reused by the bootstrap 
estimator (called on each of the cloned instances with a subsample of the data).

All attributes and methods that return a single array are reflected on the boostrap estimator in two ways: once with the same
name, in which case the mean of the estimates is returned, and again with the postfixed suffix "_interval", in which case a 
tuple of the lower and upper bounds will be returned instead (based on a 5-95% interval by default).  See the class's documentation
for more detail on how to call these methods.

.. todo::    
    * Subsampling
    * Doubly Robust Gradient Inference
