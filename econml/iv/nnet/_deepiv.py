# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Deep IV estimator and related components."""

import numpy as np
from ..._cate_estimator import BaseCateEstimator
from ...utilities import MissingModule
try:
    import torch
    import torch.nn as nn
    from torch import logsumexp, softmax, exp, log
    from torch.distributions import Distribution, Categorical, MultivariateNormal, Normal, Independent
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exn:
    torch = nn = Distribution = MissingModule(
        "torch and torch.nn are required to use the DeepIV estimator", exn)

# TODO: make sure to use random seeds wherever necessary
# TODO: make sure that the public API consistently uses "T" instead of "P" for the treatment


def _expand_dim(x, n, dim=-1):
    """Expand a single dimension by a factor of n."""
    new_size = x.size()[:dim] + (n,) + x.size()[dim:]
    return x.unsqueeze(dim).expand(new_size)


class MixtureOfGaussians(Distribution):
    """
    A mixture of Gaussians model with the specified probabilities, means, and standard deviations.

    Parameters
    ----------
    pi : tensor (n_components)
        The probabilities of each component
    mu : tensor (n_components × d_t)
        The means of each component
    sig : tensor (n_components, n_components × d_t, or n_components × d_t × d_t)
        The standard deviations of each component. If a single tensor is provided, it will be broadcast
        to the based on the number of dimensions of the means. If a 2D tensor is provided, it will be
        used as a diagonal covariance matrix. If a 3D tensor is provided, it will be
        used as a MultivariateNormal covariance matrix.
    """

    @property
    def mean(self):
        return (self.pi.unsqueeze(-2) @ self.mu).squeeze(-2)

    def __init__(self, pi, mu, sig):
        self.pi = pi
        self.mu = mu
        self.sig = sig
        if sig.dim() == mu.dim()-1:
            self.sig = sig.unsqueeze(-1).expand(mu.size())
        self.trailing_sig_dim = ()
        if sig.dim() == mu.dim()+1:
            self.trailing_sig_dim = (sig.size()[-1],)
        super().__init__(pi.size()[:-1], mu.size()[-1:])

    def sample(self, sample_shape=torch.Size()):
        ind = Categorical(self.pi).sample(sample_shape)
        if self.trailing_sig_dim:
            samples = MultivariateNormal(self.mu, self.sig).sample(sample_shape)
        else:
            samples = Independent(Normal(self.mu, self.sig), 1).sample(sample_shape)

        # expand indices to add the dimension of the gaussian
        ind = _expand_dim(ind, self.mu.size(-1), len(ind.size()))

        # add the component dimension to the index, so that it's compatible with gather
        ind = ind.unsqueeze(-2)

        # use gather to select the samples identified by ind, then squeeze out component dim
        return samples.gather(-2, ind).squeeze(-2)

    def log_prob(self, value):
        # expand value to add the component dimension
        value = value.unsqueeze(-2)
        if self.trailing_sig_dim:
            norms = MultivariateNormal(self.mu, self.sig).log_prob(value)
        else:
            norms = Independent(Normal(self.mu, self.sig), 1).log_prob(value)
        # log(sum(pi_i * N_i)) = logsumexp(log(pi_i) + log(N_i)), where log(N_i) is the log prob we already have
        return logsumexp(log(self.pi) + norms, dim=-1)


class MixtureOfGaussiansModule(nn.Module):
    def __init__(self, n_components, d_t, d_z, d_x, size_hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(d_z + d_x, size_hidden),
                                nn.ReLU(),
                                nn.Linear(size_hidden, n_components))
        self.mu = nn.Sequential(nn.Linear(d_z + d_x, size_hidden),
                                nn.ReLU(),
                                nn.Linear(size_hidden, n_components * d_t),
                                nn.Unflatten(-1, (n_components, d_t)))
        self.sig = nn.Sequential(nn.Linear(d_z + d_x, size_hidden),
                                 nn.ReLU(),
                                 nn.Linear(size_hidden, n_components))

    def forward(self, z, x):
        ft = torch.cat([z, x], dim=-1)
        return MixtureOfGaussians(softmax(self.pi(ft), dim=-1), self.mu(ft), exp(self.sig(ft)))

class DeepIV(BaseCateEstimator):
    def __init__(self, model_t, model_y,
                 batch_size,
                 n_samples, n_gradient_samples,
                 optimizer=torch.optim.Adam):
        self._model_t = model_t
        self._model_y = model_y
        self._batch_size = batch_size
        self._n_samples = n_samples
        self._n_gradient_samples = n_gradient_samples
        self._optimizer = optimizer
        super().__init__()

    def _fit_impl(self, Y, T, X, Z):
        Y, T, X, Z = [torch.from_numpy(a).float() for a in (Y, T, X, Z)]

        b = self._batch_size

        self._model_t.train()

        opt = self._optimizer(self._model_t.parameters())
        for epoch in range(100):
            total_loss = 0
            for i in range(0, len(T), b):
                opt.zero_grad()
                loss = -self._model_t(Z[i:i+b], X[i:i+b]).log_prob(T[i:i+b]).sum()
                total_loss += loss.item()
                loss.backward()
                opt.step()
            print(f"Average loss for epoch {epoch+1}: {total_loss / len(T)}")

        self._model_t.eval()

        opt = self._optimizer(self._model_y.parameters())
        for epoch in range(100):
            total_loss = 0
            for i in range(0, len(Y), b):
                x = _expand_dim(X[i:i+b], self._n_samples, dim=0)
                opt.zero_grad()
                t_s = self._model_t(Z[i:i+b], X[i:i+b]).sample((self._n_samples,))
                if self._n_gradient_samples > 0:
                    t_g = self._model_t(Z[i:i+b], X[i:i+b]).sample((self._n_gradient_samples,))
                    y_g = self._model_y(t_g, x).mean(dim=0)
                    with torch.no_grad():
                        diff = Y[i:i+b] - self._model_y(t_s, X[i:i+b].expand_as(t_s)).mean(dim=0)
                        diff_2 = diff + 2 * y_g
                    loss = (diff * (diff_2 - 2 * y_g)).sum()
                else:
                    y = _expand_dim(Y[i:i+b], self._n_samples, dim=0)
                    # no separate gradient samples; use the alternative regularizing loss from the DeepIV paper instead
                    # (take the average outside of the squared difference, rather than average predicted y inside)
                    loss = ((y - self._model_y(t_s, x)) ** 2).mean(dim=0).sum()
                total_loss += loss.item()
                loss.backward()
                opt.step()
            print(f"Average loss for epoch {epoch+1}: {total_loss / len(Y)}")

        self._model_y.eval()

    def effect(self, X=None, T0=0, T1=1):
        X, T0, T1 = [torch.from_numpy(a).float() for a in (X, T0, T1)]
        if X is None:
            X = torch.empty((len(T0), 0))
        return (self._model_y(T1, X) - self._model_y(T0, X)).detach().numpy()


class TorchResponseLoss(nn.Module):
    """
    Torch module that computes the loss of a response model on data.

    Parameters
    ----------
    h : Module (with signature (tensor, tensor) -> tensor)
        Method for generating samples of y given samples of t and x

    sample_t : int -> Tensor
        Method for getting n samples of t

    x : Tensor
        Values of x

    y : Tensor
        Values of y

    samples: int
        The number of samples to use

    use_upper_bound : bool
        Whether to use an upper bound to the true loss
        (equivalent to adding a regularization penalty on the variance of h)

    gradient_samples : int
        The number of separate additional samples to use when calculating the gradient.
        This can only be nonzero if user_upper_bound is False, in which case the gradient of
        the returned loss will be an unbiased estimate of the gradient of the true loss.

    """

    def forward(self, h, sample_t, x, y, samples=50, use_upper_bound=False, gradient_samples=50):
        assert not (use_upper_bound and gradient_samples)

        # Note that we assume that there is a single batch dimension, so that we expand x and y along dim=1
        # This is because if x or y is a vector, then expanding along dim=-2 would do the wrong thing

        # generate n samples of t, then take the mean of f(t,x) with that sample and an expanded x
        def mean(f, n):
            result = torch.mean(f(sample_t(n), _expand(x, n, dim=1)), dim=1)
            assert y.size() == result.size()
            return result

        if gradient_samples:
            # we want to separately sample the gradient; we use detach to treat the sampled model as constant
            # the overall computation ensures that we have an interpretable loss (y-h̅(p,x))²,
            # but also that the gradient is -2(y-h̅(p,x))∇h̅(p,x) with *different* samples used for each average
            diff = y - mean(h, samples)
            grad = 2 * mean(h, gradient_samples)
            return diff.detach() * ((diff + grad).detach() - grad)
        elif use_upper_bound:
            # mean of (y-h(p,x))²
            return mean(lambda t, x: (_expand(y, samples, dim=1) - h(t, x)).pow(2), samples)
        else:
            return (y - mean(h, samples)).pow(2)


class TorchDeepIVEstimator(BaseCateEstimator):
    """
    The Deep IV Estimator (see http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf).

    Parameters
    ----------
    n_components : int
        Number of components in the mixture density network

    m : Module (signature (tensor, tensor) -> tensor)
        Torch module featurizing z and x inputs

    h : Module (signature (tensor, tensor) -> tensor)
        Torch module returning y given t and x.  This should work on tensors with arbitrary leading dimensions.

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

    optimizer : parameters -> Optimizer
        The optimizer to use. Defaults to `Adam`

    inference: string, inference method, or None
        Method for performing inference.  This estimator supports 'bootstrap'
        (or an instance of `BootstrapOptions`)

    """

    def __init__(self, n_components, m, h,
                 n_samples, use_upper_bound_loss=False, n_gradient_samples=0,
                 first_stage_batch_size=32,
                 second_stage_batch_size=32,
                 first_stage_epochs=2,
                 second_stage_epochs=2,
                 optimizer=torch.optim.Adam,
                 inference=None):
        self._n_components = n_components
        self._m = m
        self._h = h
        self._n_samples = n_samples
        self._use_upper_bound_loss = use_upper_bound_loss
        self._n_gradient_samples = n_gradient_samples
        self._first_stage_batch_size = first_stage_batch_size
        self._second_stage_batch_size = second_stage_batch_size
        self._first_stage_epochs = first_stage_epochs
        self._second_stage_epochs = second_stage_epochs
        self._optimizer = optimizer
        super().__init__(inference=inference)

    def _fit_impl(self, Y, T, X, Z):
        """Estimate the counterfactual model from data.

        That is, estimate functions τ(·, ·, ·), ∂τ(·, ·).

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: optional (n × dₓ) matrix
            Features for each sample
        Z: optional (n × d_z) matrix
            Instruments for each sample

        Returns
        -------
        self

        """
        assert 1 <= np.ndim(X) <= 2
        assert 1 <= np.ndim(Z) <= 2
        assert 1 <= np.ndim(T) <= 2
        assert 1 <= np.ndim(Y) <= 2
        assert np.shape(X)[0] == np.shape(Y)[0] == np.shape(T)[0] == np.shape(Z)[0]
        # in case vectors were passed for Y or T, keep track of trailing dims for reshaping effect output

        d_x, d_y, d_z, d_t = [np.shape(a)[1:] for a in [X, Y, Z, T]]
        self._d_y = d_y

        d_m = self._m(torch.Tensor(np.empty((1,) + d_z)), torch.Tensor(np.empty((1,) + d_x))).size()[1]

        Y, T, X, Z = [torch.from_numpy(A).float() for A in (Y, T, X, Z)]
        n_components = self._n_components

        treatment_model = self._m

        class Mog(nn.Module):
            def __init__(self):
                super().__init__()
                self.treatment_model = treatment_model
                self.mog_model = MixtureOfGaussians(n_components, d_m, (d_t if d_t else None))

            def forward(self, z, x):
                features = self.treatment_model(z, x)
                return self.mog_model(features)

        mog = Mog()
        self._mog = mog
        mog.train()
        opt = self._optimizer(mog.parameters())

        # train first-stage model
        loader = DataLoader(TensorDataset(T, Z, X), shuffle=True, batch_size=self._first_stage_batch_size)
        for epoch in range(self._first_stage_epochs):
            total_loss = 0
            for i, (t, z, x) in enumerate(loader):
                opt.zero_grad()
                pi, mu, sig = mog(z, x)
                loss = TorchMogLoss()(pi, mu, sig, t).sum()
                total_loss += loss.item()
                if i % 30 == 0:
                    print(loss / t.size()[0])
                loss.backward()
                opt.step()
            print(f"Average loss for epoch {epoch+1}: {total_loss / len(loader.dataset)}")

        mog.eval()  # set mog to evaluation mode
        for p in mog.parameters():
            p.requires_grad_(False)

        self._h.train()
        opt = self._optimizer(self._h.parameters())

        loader = DataLoader(TensorDataset(Y, Z, X), shuffle=True, batch_size=self._second_stage_batch_size)
        for epoch in range(self._second_stage_epochs):
            total_loss = 0
            for i, (y, z, x) in enumerate(loader):
                opt.zero_grad()
                pi, mu, sig = mog(z, x)
                loss = TorchResponseLoss()(self._h,
                                           lambda n: TorchMogSampleModel()(n, pi, mu, sig),
                                           x, y,
                                           self._n_samples, self._use_upper_bound_loss, self._n_gradient_samples)
                loss = loss.sum()
                total_loss += loss.item()
                if i % 30 == 0:
                    print(loss / y.size()[0])
                loss.backward()
                opt.step()
            print(f"Average loss for epoch {epoch+1}: {total_loss / len(loader.dataset)}")

        self._h.eval()  # set h to evaluation mode
        for p in self._h.parameters():
            p.requires_grad_(False)

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
        X: optional (m × dₓ) matrix
            Features for each sample

        Returns
        -------
        τ: (m × d_y) matrix
            Heterogeneous treatment effects on each outcome for each sample
            Note that when Y is a vector rather than a 2-dimensional array, the corresponding
            singleton dimension will be collapsed (so this method will return a vector)
        """
        if np.ndim(T0) == 0:
            T0 = np.repeat(T0, 1 if X is None else np.shape(X)[0])
        if np.ndim(T1) == 0:
            T1 = np.repeat(T1, 1 if X is None else np.shape(X)[0])
        if X is None:
            X = np.empty((np.shape(T0)[0], 0))
        return self.predict(T1, X) - self.predict(T0, X)

    def marginal_effect(self, T, X=None):
        """
        Calculate the marginal effect ∂τ(·, ·) around a base treatment point conditional on features.

        Parameters
        ----------
        T: (m × dₜ) matrix
            Base treatments for each sample
        X: optional(m × dₓ) matrix
            Features for each sample

        Returns
        -------
        grad_tau: (m × d_y × dₜ) array
            Heterogeneous marginal effects on each outcome for each sample
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        if X is None:
            X = np.empty((np.shape(T)[0], 0))
        X, T = [torch.from_numpy(A).float() for A in [X, T]]
        if self._d_y:
            X, T = [A.unsqueeze(1).expand((-1,) + self._d_y + (-1,)) for A in [X, T]]
        T.requires_grad_(True)
        if self._d_y:
            self._h(T, X).backward(torch.eye(self._d_y[0]).expand(X.size()[0], -1, -1))
            return T.grad.numpy()
        else:
            self._h(T, X).backward(torch.ones(X.size()[0]))
            return T.grad.numpy()

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
        X, T = [torch.from_numpy(A).float() for A in [X, T]]
        return self._h(T, X).numpy().reshape((-1,) + self._d_y)
