# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Deep IV estimator and related components."""

import numpy as np
from ..._cate_estimator import BaseCateEstimator
from ...utilities import check_input_arrays, _deprecate_positional, deprecated, MissingModule
try:
    import keras
    from keras import backend as K
    import keras.layers as L
    from keras.models import Model
except ImportError as exn:
    keras = K = L = Model = MissingModule("keras and tensorflow are no longer dependencies of the main econml "
                                          "package; install econml[tf] or econml[all] to require them, or install "
                                          "them separately, to use DeepIV", exn)

# TODO: make sure to use random seeds wherever necessary
# TODO: make sure that the public API consistently uses "T" instead of "P" for the treatment

# unfortunately with the Theano and Tensorflow backends,
# the straightforward use of K.stop_gradient can cause an error
# because the parameters of the intermediate layers are now disconnected from the loss;
# therefore we add a pointless multiplication by 0 to the values in each of the variables in vs
# so that those layers remain connected but with 0 gradient


def _zero_grad(e, vs):
    if K.backend() == 'cntk':
        return K.stop_gradient(e)
    else:
        z = 0 * K.sum(K.concatenate([K.batch_flatten(v) for v in vs]))
        return K.stop_gradient(e) + z


def mog_model(n_components, d_x, d_t):
    """
    Create a mixture of Gaussians model with the specified number of components.

    Parameters
    ----------
    n_components : int
        The number of components in the mixture model

    d_x : int
        The number of dimensions in the layer used as input

    d_t : int
        The number of dimensions in the output

    Returns
    -------
    A Keras model that takes an input of dimension `d_t` and generates three outputs: pi, mu, and sigma

    """
    x = L.Input((d_x,))
    pi = L.Dense(n_components, activation='softmax')(x)
    mu = L.Reshape((n_components, d_t))(L.Dense(n_components * d_t)(x))
    log_sig = L.Dense(n_components)(x)
    sig = L.Lambda(K.exp)(log_sig)
    return Model([x], [pi, mu, sig])


def mog_loss_model(n_components, d_t):
    """
    Create a Keras model that computes the loss of a mixture of Gaussians model on data.

    Parameters
    ----------
    n_components : int
        The number of components in the mixture model

    d_t : int
        The number of dimensions in the output

    Returns
    -------
    A Keras model that takes as inputs pi, mu, sigma, and t and generates a single output containing the loss.

    """
    pi = L.Input((n_components,))
    mu = L.Input((n_components, d_t))
    sig = L.Input((n_components,))
    t = L.Input((d_t,))

    # || t - mu_i || ^2
    d2 = L.Lambda(lambda d: K.sum(K.square(d), axis=-1),
                  output_shape=(n_components,))(
        L.Subtract()([L.RepeatVector(n_components)(t), mu])
    )

    # LL = C - log(sum(pi_i/sig^d * exp(-d2/(2*sig^2))))
    # Use logsumexp for numeric stability:
    # LL = C - log(sum(exp(-d2/(2*sig^2) + log(pi_i/sig^d))))
    # TODO: does the numeric stability actually make any difference?
    def make_logloss(d2, sig, pi):
        return -K.logsumexp(-d2 / (2 * K.square(sig)) + K.log(pi / K.pow(sig, d_t)), axis=-1)

    ll = L.Lambda(lambda dsp: make_logloss(*dsp), output_shape=(1,))([d2, sig, pi])

    m = Model([pi, mu, sig, t], [ll])
    return m


def mog_sample_model(n_components, d_t):
    """
    Create a model that generates samples from a mixture of Gaussians.

    Parameters
    ----------
    n_components : int
        The number of components in the mixture model

    d_t : int
        The number of dimensions in the output

    Returns
    -------
    A Keras model that takes as inputs pi, mu, and sigma, and generates a single output containing a sample.

    """
    pi = L.Input((n_components,))
    mu = L.Input((n_components, d_t))
    sig = L.Input((n_components,))

    # CNTK backend can't randomize across batches and doesn't implement cumsum (at least as of June 2018,
    # see Known Issues on https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras)
    def sample(pi, mu, sig):
        batch_size = K.shape(pi)[0]
        if K.backend() == 'cntk':
            # generate cumulative sum via matrix multiplication
            cumsum = K.dot(pi, K.constant(np.triu(np.ones((n_components, n_components)))))
        else:
            cumsum = K.cumsum(pi, 1)
        cumsum_shift = K.concatenate([K.zeros_like(cumsum[:, 0:1]), cumsum])[:, :-1]
        if K.backend() == 'cntk':
            import cntk as C
            # Generate standard uniform values in shape (batch_size,1)
            #   (since we can't use the dynamic batch_size with random.uniform in CNTK,
            #    we use uniform_like instead with an input of an appropriate shape)
            rndSmp = C.random.uniform_like(pi[:, 0:1])
        else:
            rndSmp = K.random_uniform((batch_size, 1))
        cmp1 = K.less_equal(cumsum_shift, rndSmp)
        cmp2 = K.less(rndSmp, cumsum)

        # convert to floats and multiply to perform equivalent of logical AND
        rndIndex = K.cast(cmp1, K.floatx()) * K.cast(cmp2, K.floatx())

        if K.backend() == 'cntk':
            # Generate standard normal values in shape (batch_size,1,d_t)
            #   (since we can't use the dynamic batch_size with random.normal in CNTK,
            #    we use normal_like instead with an input of an appropriate shape)
            rndNorms = C.random.normal_like(mu[:, 0:1, :])  # K.random_normal((1,d_t))
        else:
            rndNorms = K.random_normal((batch_size, 1, d_t))

        rndVec = mu + K.expand_dims(sig) * rndNorms

        # exactly one entry should be nonzero for each b,d combination; use sum to select it
        return K.sum(K.expand_dims(rndIndex) * rndVec, 1)

    # prevent gradient from passing through sampling
    samp = L.Lambda(lambda pms: _zero_grad(sample(*pms), pms), output_shape=(d_t,))
    samp.trainable = False

    return Model([pi, mu, sig], samp([pi, mu, sig]))


# three options: biased or upper-bound loss require a single number of samples;
#                unbiased can take different numbers for the network and its gradient
def response_loss_model(h, p, d_z, d_x, d_y, samples=1, use_upper_bound=False, gradient_samples=0):
    """
    Create a Keras model that computes the loss of a response model on data.

    Parameters
    ----------
    h : (tensor, tensor) -> Layer
        Method for building a model of y given p and x

    p : (tensor, tensor) -> Layer
        Method for building a model of p given z and x

    d_z : int
        The number of dimensions in z

    d_x :  int
        Tbe number of dimensions in x

    d_y : int
        The number of dimensions in y

    samples: int
        The number of samples to use

    use_upper_bound : bool
        Whether to use an upper bound to the true loss
        (equivalent to adding a regularization penalty on the variance of h)

    gradient_samples : int
        The number of separate additional samples to use when calculating the gradient.
        This can only be nonzero if user_upper_bound is False, in which case the gradient of
        the returned loss will be an unbiased estimate of the gradient of the true loss.

    Returns
    -------
    A Keras model that takes as inputs z, x, and y and generates a single output containing the loss.

    """
    assert not (use_upper_bound and gradient_samples)

    # sample: (() -> Layer, int) -> Layer
    def sample(f, n):
        assert n > 0
        if n == 1:
            return f()
        else:
            return L.average([f() for _ in range(n)])
    z, x, y = [L.Input((d,)) for d in [d_z, d_x, d_y]]
    if gradient_samples:
        # we want to separately sample the gradient; we use stop_gradient to treat the sampled model as constant
        # the overall computation ensures that we have an interpretable loss (y-h̅(p,x))²,
        # but also that the gradient is -2(y-h̅(p,x))∇h̅(p,x) with *different* samples used for each average
        diff = L.subtract([y, sample(lambda: h(p(z, x), x), samples)])
        grad = sample(lambda: h(p(z, x), x), gradient_samples)

        def make_expr(grad, diff):
            return K.stop_gradient(diff) * (K.stop_gradient(diff + 2 * grad) - 2 * grad)
        expr = L.Lambda(lambda args: make_expr(*args))([grad, diff])
    elif use_upper_bound:
        expr = sample(lambda: L.Lambda(K.square)(L.subtract([y, h(p(z, x), x)])), samples)
    else:
        expr = L.Lambda(K.square)(L.subtract([y, sample(lambda: h(p(z, x), x), samples)]))
    return Model([z, x, y], [expr])


class DeepIV(BaseCateEstimator):
    """
    The Deep IV Estimator (see http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf).

    Parameters
    ----------
    n_components : int
        Number of components in the mixture density network

    m : (tensor, tensor) -> Layer
        Method for building a Keras model that featurizes the z and x inputs

    h : (tensor, tensor) -> Layer
        Method for building a model of y given t and x

    n_samples : int
        The number of samples to use

    use_upper_bound_loss : bool, optional
        Whether to use an upper bound to the true loss
        (equivalent to adding a regularization penalty on the variance of h).
        Defaults to False.

    n_gradient_samples : int, optional
        The number of separate additional samples to use when calculating the gradient.
        This can only be nonzero if user_upper_bound is False, in which case the gradient of
        the returned loss will be an unbiased estimate of the gradient of the true loss.
        Defaults to 0.

    optimizer : str, optional
        The optimizer to use. Defaults to "adam"

    first_stage_options : dictionary, optional
        The keyword arguments to pass to Keras's `fit` method when training the first stage model.
        Defaults to `{"epochs": 100}`.

    second_stage_options : dictionary, optional
        The keyword arguments to pass to Keras's `fit` method when training the second stage model.
        Defaults to `{"epochs": 100}`.

    """

    def __init__(self, *,
                 n_components,
                 m,
                 h,
                 n_samples, use_upper_bound_loss=False, n_gradient_samples=0,
                 optimizer='adam',
                 first_stage_options={"epochs": 100},
                 second_stage_options={"epochs": 100}):
        self._n_components = n_components
        self._m = m
        self._h = h
        self._n_samples = n_samples
        self._use_upper_bound_loss = use_upper_bound_loss
        self._n_gradient_samples = n_gradient_samples
        self._optimizer = optimizer
        self._first_stage_options = first_stage_options
        self._second_stage_options = second_stage_options
        super().__init__()

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, *, X, Z, inference=None):
        """Estimate the counterfactual model from data.

        That is, estimate functions τ(·, ·, ·), ∂τ(·, ·).

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: (n × dₓ) matrix
            Features for each sample
        Z: (n × d_z) matrix
            Instruments for each sample
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`)

        Returns
        -------
        self

        """
        Y, T, X, Z = check_input_arrays(Y, T, X, Z)
        assert 1 <= np.ndim(X) <= 2
        assert 1 <= np.ndim(Z) <= 2
        assert 1 <= np.ndim(T) <= 2
        assert 1 <= np.ndim(Y) <= 2
        assert np.shape(X)[0] == np.shape(Y)[0] == np.shape(T)[0] == np.shape(Z)[0]

        # in case vectors were passed for Y or T, keep track of trailing dims for reshaping effect output

        d_x, d_y, d_z, d_t = [np.shape(a)[1] if np.ndim(a) > 1 else 1 for a in [X, Y, Z, T]]
        x_in, y_in, z_in, t_in = [L.Input((d,)) for d in [d_x, d_y, d_z, d_t]]
        n_components = self._n_components

        treatment_network = self._m(z_in, x_in)

        # the dimensionality of the output of the network
        # TODO: is there a more robust way to do this?
        d_n = K.int_shape(treatment_network)[-1]

        pi, mu, sig = mog_model(n_components, d_n, d_t)([treatment_network])

        ll = mog_loss_model(n_components, d_t)([pi, mu, sig, t_in])

        model = Model([z_in, x_in, t_in], [ll])
        model.add_loss(L.Lambda(K.mean)(ll))
        model.compile(self._optimizer)
        # TODO: do we need to give the user more control over other arguments to fit?
        model.fit([Z, X, T], [], **self._first_stage_options)

        lm = response_loss_model(lambda t, x: self._h(t, x),
                                 lambda z, x: Model([z_in, x_in],
                                                    # subtle point: we need to build a new model each time,
                                                    # because each model encapsulates its randomness
                                                    [mog_sample_model(n_components, d_t)([pi, mu, sig])])([z, x]),
                                 d_z, d_x, d_y,
                                 self._n_samples, self._use_upper_bound_loss, self._n_gradient_samples)

        rl = lm([z_in, x_in, y_in])
        response_model = Model([z_in, x_in, y_in], [rl])
        response_model.add_loss(L.Lambda(K.mean)(rl))
        response_model.compile(self._optimizer)
        # TODO: do we need to give the user more control over other arguments to fit?
        response_model.fit([Z, X, Y], [], **self._second_stage_options)

        self._effect_model = Model([t_in, x_in], [self._h(t_in, x_in)])

        # TODO: it seems like we need to sum over the batch because we can only apply gradient to a scalar,
        #       not a general tensor (because of how backprop works in every framework)
        #       (alternatively, we could iterate through the batch in addition to iterating through the output,
        #       but this seems annoying...)
        #       Therefore, it's important that we use a batch size of 1 when we call predict with this model
        def calc_grad(t, x):
            h = self._h(t, x)
            all_grads = K.concatenate([g
                                       for i in range(d_y)
                                       for g in K.gradients(K.sum(h[:, i]), [t])])
            return K.reshape(all_grads, (-1, d_y, d_t))

        self._marginal_effect_model = Model([t_in, x_in], L.Lambda(lambda tx: calc_grad(*tx))([t_in, x_in]))

    def effect(self, X=None, T0=0, T1=1):
        """
        Calculate the heterogeneous treatment effect τ(·,·,·).

        The effect is calculated between the two treatment points
        conditional on a vector of features on a set of m test samples {T0ᵢ, T1ᵢ, Xᵢ}.

        Parameters
        ----------
        T0: (m × dₜ) matrix
            Base treatments for each sample
        T1: (m × dₜ) matrix
            Target treatments for each sample
        X:  (m × dₓ) matrix, optional
            Features for each sample

        Returns
        -------
        τ: (m × d_y) matrix
            Heterogeneous treatment effects on each outcome for each sample
            Note that when Y is a vector rather than a 2-dimensional array, the corresponding
            singleton dimension will be collapsed (so this method will return a vector)
        """
        X, T0, T1 = check_input_arrays(X, T0, T1)
        if np.ndim(T0) == 0:
            T0 = np.repeat(T0, 1 if X is None else np.shape(X)[0])
        if np.ndim(T1) == 0:
            T1 = np.repeat(T1, 1 if X is None else np.shape(X)[0])
        if X is None:
            X = np.empty((np.shape(T0)[0], 0))
        return (self._effect_model.predict([T1, X]) - self._effect_model.predict([T0, X])).reshape((-1,) + self._d_y)

    def marginal_effect(self, T, X=None):
        """
        Calculate the marginal effect ∂τ(·, ·) around a base treatment point conditional on features.

        Parameters
        ----------
        T: (m × dₜ) matrix
            Base treatments for each sample
        X: (m × dₓ) matrix, optional
            Features for each sample

        Returns
        -------
        grad_tau: (m × d_y × dₜ) array
            Heterogeneous marginal effects on each outcome for each sample
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        T, X = check_input_arrays(T, X)
        # TODO: any way to get this to work on batches of arbitrary size?
        return self._marginal_effect_model.predict([T, X], batch_size=1).reshape((-1,) + self._d_y + self._d_t)

    def predict(self, T, X):
        """Predict outcomes given treatment assignments and features.

        Parameters
        ----------
        T: (m × dₜ) matrix
            Base treatments for each sample
        X: (m × dₓ) matrix
            Features for each sample

        Returns
        -------
        Y: (m × d_y) matrix
            Outcomes for each sample
            Note that when Y is a vector rather than a 2-dimensional array, the corresponding
            singleton dimension will be collapsed (so this method will return a vector)
        """
        T, X = check_input_arrays(T, X)
        return self._effect_model.predict([T, X]).reshape((-1,) + self._d_y)
