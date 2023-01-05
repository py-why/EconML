import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from scipy.special import expit

_ihdp_sim_file = os.path.join(os.path.dirname(__file__), "ihdp", "sim.csv")
_ihdp_sim_data = pd.read_csv(_ihdp_sim_file)


def ihdp_surface_A(random_state=None):
    """ Generates semi-synthetic, constant treatment effect data according to response surface A
        from Hill (2011).

        Parameters
        ----------
        random_state : int, RandomState instance, or None, default None
                If int, random_state is the seed used by the random number generator;
            If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
            If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
            by :mod:`np.random<numpy.random>`.

        Returns
        -------
        Y : array_like, shape (n, d_y)
            Outcome for the treatment policy.

        T : array_like, shape (n, d_t)
            Binary treatment policy.

        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.
    """
    # Remove children with nonwhite mothers from the treatment group
    T, X = _process_ihdp_sim_data()
    n = X.shape[0]
    d_x = X.shape[1]
    random_state = check_random_state(random_state)
    beta = random_state.choice([0, 1, 2, 3, 4], size=d_x, replace=True, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    Y = np.dot(X, beta) + T * 4 + random_state.normal(0, 1, size=n)
    true_TE = np.ones(X.shape[0]) * 4
    return Y, T, X, true_TE


def ihdp_surface_B(random_state=None):
    """ Generates semi-synthetic, heterogeneous treatment effect data according to response surface B
        from Hill (2011).

        Parameters
        ----------
        random_state : int, RandomState instance, or None, default None
                If int, random_state is the seed used by the random number generator;
            If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
            If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
            by :mod:`np.random<numpy.random>`.

        Returns
        -------
        Y : array_like, shape (n, d_y)
            Outcome for the treatment policy.

        T : array_like, shape (n, d_t)
            Binary treatment policy.

        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.
    """
    T, X = _process_ihdp_sim_data()
    n = X.shape[0]
    d_x = X.shape[1]
    random_state = check_random_state(random_state)
    beta = random_state.choice([0, 0.1, 0.2, 0.3, 0.4], size=d_x, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])
    offset = np.concatenate((np.zeros((n, 1)), np.ones((n, d_x - 1)) * 0.5), axis=1)
    omega = np.mean((np.dot(X, beta) - np.exp(np.dot(X + offset, beta)))[T == 1]) - 4
    Y = (np.dot(X, beta) - omega) * T + np.exp(np.dot(X + offset, beta)) * (1 - T) + random_state.normal(0, 1, size=n)
    true_TE = ((np.dot(X, beta) - omega) - np.exp(np.dot(X + offset, beta)))
    return Y, T, X, true_TE


def _process_ihdp_sim_data():
    # Remove children with nonwhite mothers from the treatment group
    data_subset = _ihdp_sim_data[~((_ihdp_sim_data['treat'] == 1) & (_ihdp_sim_data['momwhite'] == 0))]
    T = data_subset['treat'].values
    # Select columns
    X = data_subset[['bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage',
                     'sex', 'twin', 'b.marr', 'mom.lths', 'mom.hs', 'mom.scoll', 'cig',
                     'first', 'booze', 'drugs', 'work.dur', 'prenatal', 'site1', 'site2',
                     'site3', 'site4', 'site5', 'site6', 'site7']].values
    # Scale the numeric variables
    X[:, :6] = StandardScaler().fit_transform(X[:, :6])
    # Change the binary variable 'first' takes values in {1,2}
    X[:, 13] = X[:, 13] + 1
    # Append a column of ones as intercept
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    return T, X


class StandardDGP:
    """
    A class to generate synthetic causal datasets

    Parameters
    ----------
    n: int
        Number of observations to generate

    d_t: int
        Dimensionality of treatment

    d_y: int
        Dimensionality of outcome

    d_x: int
        Dimensionality of features

    d_z: int
        Dimensionality of instrument

    discrete_treatment: bool
        Dimensionality of treatment

    discrete_isntrument: bool
        Dimensionality of instrument

    squeeze_T: bool
        Whether to squeeze the final T array on output

    squeeze_Y: bool
        Whether to squeeze the final Y array on output

    nuisance_Y: func or dict
        Nuisance function. Describes how the covariates affect the outcome. 
        If a function, this function will be used on features X to partially generate Y. 
        If a dict, must include 'support' and 'degree' keys.

    nuisance_T: func or dict
        Nuisance function. Describes how the covariates affect the treatment.
        If a function, this function will be used on features X to partially generate T. 
        If a dict, must include 'support' and 'degree' keys.

    nuisance_TZ: func or dict
        Nuisance function. Describes how the instrument affects the treatment.
        If a function, this function will be used on instrument Z to partially generate T. 
        If a dict, must include 'support' and 'degree' keys.

    theta: func or dict
        Describes how the features affects the treatment effect heterogenity.
        If a function, this function will be used on features X to calculate treatment effect heterogenity. 
        If a dict, must include 'support' and 'degree' keys.

    y_of_t: func or dict
        Describes how the treatment affects the outcome.
        If a function, this function will be used directly. 
        If a dict, must include 'support' and 'degree' keys.

    x_eps: float
        Noise parameter for feature generation

    y_eps: func or dict
        Noise parameter for outcome generation

    t_eps: func or dict
        Noise parameter for treatment generation

    """
    def __init__(self,
                 n=1000,
                 d_t=1,
                 d_y=1,
                 d_x=5,
                 d_z=None,
                 discrete_treatment=False,
                 discrete_instrument=False,
                 squeeze_T=False,
                 squeeze_Y=False,
                 nuisance_Y=None,
                 nuisance_T=None,
                 nuisance_TZ=None,
                 theta=None,
                 y_of_t=None,
                 x_eps=1,
                 y_eps=1,
                 t_eps=1,
                 random_state=None
                 ):
        self.n = n
        self.d_t = d_t
        self.d_y = d_y
        self.d_x = d_x
        self.d_z = d_z

        self.discrete_treatment = discrete_treatment
        self.discrete_instrument = discrete_instrument
        self.squeeze_T = squeeze_T
        self.squeeze_Y = squeeze_Y

        if callable(nuisance_Y):
            self.nuisance_Y = nuisance_Y
        else:  # else must be dict
            if nuisance_Y is None:
                nuisance_Y = {'support': self.d_x, 'degree': 1}
            assert isinstance(nuisance_Y, dict), f"nuisance_Y must be a callable or dict, but got {type(nuisance_Y)}"
            self.nuisance_Y, self.nuisance_Y_coefs = self.gen_nuisance(**nuisance_Y)

        if callable(nuisance_T):
            self.nuisance_T = nuisance_T
        else:  # else must be dict
            if nuisance_T is None:
                nuisance_T = {'support': self.d_x, 'degree': 1}
            assert isinstance(nuisance_T, dict), f"nuisance_T must be a callable or dict, but got {type(nuisance_T)}"
            self.nuisance_T, self.nuisance_T_coefs = self.gen_nuisance(**nuisance_T)

        if self.d_z:
            if callable(nuisance_TZ):
                self.nuisance_TZ = nuisance_TZ
            else:  # else must be dict
                if nuisance_TZ is None:
                    nuisance_TZ = {'support': self.d_z, 'degree': 1}
                assert isinstance(
                    nuisance_TZ, dict), f"nuisance_TZ must be a callable or dict, but got {type(nuisance_TZ)}"
                nuisance_TZ = {**nuisance_TZ, 'k': self.d_z}
                self.nuisance_TZ, self.nuisance_TZ_coefs = self.gen_nuisance(**nuisance_TZ)
        else:
            self.nuisance_TZ = lambda x: 0

        if callable(theta):
            self.theta = theta
        else:  # else must be dict
            if theta is None:
                theta = {'support': self.d_x, 'degree': 1, 'bounds': [1, 2], 'intercept': [1, 2]}
            assert isinstance(theta, dict), f"theta must be a callable or dict, but got {type(theta)}"
            self.theta, self.theta_coefs = self.gen_nuisance(**theta)

        if callable(y_of_t):
            self.y_of_t = y_of_t
        else:  # else must be dict
            if y_of_t is None:
                y_of_t = {'support': self.d_t, 'degree': 1, 'bounds': [1, 1]}
            assert isinstance(y_of_t, dict), f"y_of_t must be a callable or dict, but got {type(y_of_t)}"
            y_of_t['k'] = self.d_t
            self.y_of_t, self.y_of_t_coefs = self.gen_nuisance(**y_of_t)

        self.x_eps = x_eps
        self.y_eps = y_eps
        self.t_eps = t_eps

    def gen_Y(self):
        self.y_noise = np.random.normal(size=(self.n, self.d_y), scale=self.y_eps)
        self.Y = self.theta(self.X) * self.y_of_t(self.T) + self.nuisance_Y(self.X) + self.y_noise
        return self.Y

    def gen_X(self):
        self.X = np.random.normal(size=(self.n, self.d_x), scale=self.x_eps)
        return self.X

    def gen_T(self):
        noise = np.random.normal(size=(self.n, self.d_t), scale=self.t_eps)
        self.T_noise = noise

        if self.discrete_treatment:
            prob_T = expit(self.nuisance_T(self.X) + self.nuisance_TZ(self.Z) + self.T_noise)
            self.T = np.random.binomial(1, prob_T)
            return self.T

        else:
            self.T = self.nuisance_T(self.X) + self.nuisance_TZ(self.Z) + self.T_noise
            return self.T

    def gen_Z(self):
        if self.d_z:
            if self.discrete_instrument:
                self.Z = np.random.binomial(1, 0.5, size=(self.n, self.d_z))
                return self.Z

            else:
                Z_noise = np.random.normal(size=(self.n, self.d_z), loc=3, scale=3)
                self.Z = Z_noise
                return self.Z

        else:
            self.Z = None
            return self.Z

    def gen_nuisance(self, k=None, support=1, bounds=[-1, 1], degree=1, intercept=False):
        """
        A function to generate nuisance functions. Returns a nuisance function and corresponding coefs. 

        Parameters
        ----------
        k: int
            Dimension of input for nuisance function

        support: int
            Number of non-zero coefficients

        bounds: int list
            Bounds for coefficients which will be generated uniformly. Represented as [low, high]

        degree: int
            Input will be raised to this degree before multiplying with coefficients

        intercept:
            Bounds for intercept which will be generated uniformly. Represented as [low, high]
        """
        if not k:
            k = self.d_x

        coefs = np.random.uniform(low=bounds[0], high=bounds[1], size=k)
        supports = np.random.choice(k, size=support, replace=False)
        mask = np.zeros(shape=k)
        mask[supports] = 1
        coefs = coefs * mask

        orders = np.ones(shape=(k,)) * degree  # enforce all to be the degree for now

        if intercept:
            assert len(intercept)==2, 'intercept must be a list of 2 numbers, representing lower and upper bounds'
            intercept = np.random.uniform(low=intercept[0], high=intercept[1])
        else:
            intercept = 0

        def calculate_nuisance(W):
            W2 = np.copy(W)
            for i in range(0, k):
                W2[:, i] = W[:, i]**orders[i]
            out = W2.dot(coefs)
            return out.reshape(-1, 1) + intercept

        return calculate_nuisance, coefs

    def effect(self, X, T0, T1):
        if T0 is None or T0 == 0:
            T0 = np.zeros(shape=(T1.shape[0], self.d_t))

        effect_t1 = self.theta(X) * self.y_of_t(T1)
        effect_t0 = self.theta(X) * self.y_of_t(T0)
        return effect_t1 - effect_t0

    def const_marginal_effect(self, X):
        return self.theta(X)

    def gen_data(self):
        X = self.gen_X()
        Z = self.gen_Z()
        T = self.gen_T()
        Y = self.gen_Y()

        if self.squeeze_T:
            T = T.squeeze()
        if self.squeeze_Y:
            Y = Y.squeeze()

        data_dict = {
            'Y': Y,
            'T': T,
            'X': X
        }

        if self.d_z:
            data_dict['Z'] = Z

        return data_dict
