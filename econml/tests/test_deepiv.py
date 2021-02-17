# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for `deepiv` module."""

import unittest
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder

import keras
import keras.backend as K

import pytest

from econml.iv.nnet._deepiv import _zero_grad
from econml.iv.nnet import DeepIV
from econml.iv.nnet._deepiv import mog_model, mog_loss_model, mog_sample_model, response_loss_model
from econml.utilities import reshape


class TestDeepIV(unittest.TestCase):
    def test_stop_grad(self):
        x_input = keras.layers.Input(shape=(1,))
        z_input = keras.layers.Input(shape=(1,))
        y_input = keras.layers.Input(shape=(1,))
        x_intermediate = keras.layers.Dense(1)(x_input)
        x = keras.layers.Dense(1, trainable=False)(x_intermediate)
        sum = keras.layers.Lambda(lambda xz: _zero_grad(xz[0], [xz[0]]) + xz[1])([x, z_input])
        loss = keras.layers.Lambda(K.square)(keras.layers.subtract([y_input, sum]))

        model = keras.Model([x_input, y_input, z_input], [loss])
        model.add_loss(K.mean(loss))
        model.compile('nadam')
        model.fit([np.array([[1]]), np.array([[2]]), np.array([[0]])], [])

    @pytest.mark.slow
    def test_deepiv_shape(self):
        fit_opts = {"epochs": 2}

        """Make sure that arbitrary sizes for t, z, x, and y don't break the basic operations."""
        for _ in range(5):
            d_t = np.random.choice(range(1, 4))  # number of treatments
            d_z = np.random.choice(range(1, 4))  # number of instruments
            d_x = np.random.choice(range(1, 4))  # number of features
            d_y = np.random.choice(range(1, 4))  # number of responses
            n = 500
            # simple DGP only for illustration
            x = np.random.uniform(size=(n, d_x))
            z = np.random.uniform(size=(n, d_z))
            p_x_t = np.random.uniform(size=(d_x, d_t))
            p_z_t = np.random.uniform(size=(d_z, d_t))
            t = x @ p_x_t + z @ p_z_t
            p_xt_y = np.random.uniform(size=(d_x * d_t, d_y))
            y = (x.reshape(n, -1, 1) * t.reshape(n, 1, -1)).reshape(n, -1) @ p_xt_y

            # Define the treatment model neural network architecture
            # This will take the concatenation of one-dimensional values z and x as input,
            # so the input shape is (d_z + d_x,)
            # The exact shape of the final layer is not critical because the Deep IV framework will
            # add extra layers on top for the mixture density network
            treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(d_z + d_x,)),
                                                keras.layers.Dropout(0.17),
                                                keras.layers.Dense(64, activation='relu'),
                                                keras.layers.Dropout(0.17),
                                                keras.layers.Dense(32, activation='relu'),
                                                keras.layers.Dropout(0.17)])

            # Define the response model neural network architecture
            # This will take the concatenation of one-dimensional values t and x as input,
            # so the input shape is (d_t + d_x,)
            # The output should match the shape of y, so it must have shape (d_y,) in this case
            # NOTE: For the response model, it is important to define the model *outside*
            #       of the lambda passed to the DeepIvEstimator, as we do here,
            #       so that the same weights will be reused in each instantiation
            response_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(d_t + d_x,)),
                                               keras.layers.Dropout(0.17),
                                               keras.layers.Dense(64, activation='relu'),
                                               keras.layers.Dropout(0.17),
                                               keras.layers.Dense(32, activation='relu'),
                                               keras.layers.Dropout(0.17),
                                               keras.layers.Dense(d_y)])

            deepIv = DeepIV(n_components=10,  # number of gaussians in our mixture density network
                            m=lambda z, x: treatment_model(
                                keras.layers.concatenate([z, x])),  # treatment model
                            h=lambda t, x: response_model(keras.layers.concatenate([t, x])),  # response model
                            n_samples=1,  # number of samples to use to estimate the response
                            use_upper_bound_loss=False,  # whether to use an approximation to the true loss
                            # number of samples to use in second estimate of the response
                            # (to make loss estimate unbiased)
                            n_gradient_samples=1,
                            # Keras optimizer to use for training - see https://keras.io/optimizers/
                            optimizer='adam',
                            first_stage_options=fit_opts,
                            second_stage_options=fit_opts)

            deepIv.fit(Y=y, T=t, X=x, Z=z)
            # do something with predictions...
            deepIv.predict(T=t, X=x)
            deepIv.effect(x, np.zeros_like(t), t)

        # also test vector t and y
        for _ in range(3):
            d_z = np.random.choice(range(1, 4))  # number of instruments
            d_x = np.random.choice(range(1, 4))  # number of features
            n = 500
            # simple DGP only for illustration
            x = np.random.uniform(size=(n, d_x))
            z = np.random.uniform(size=(n, d_z))
            p_x_t = np.random.uniform(size=(d_x,))
            p_z_t = np.random.uniform(size=(d_z,))
            t = x @ p_x_t + z @ p_z_t
            p_xt_y = np.random.uniform(size=(d_x,))
            y = (x * t.reshape(n, 1)) @ p_xt_y

            # Define the treatment model neural network architecture
            # This will take the concatenation of one-dimensional values z and x as input,
            # so the input shape is (d_z + d_x,)
            # The exact shape of the final layer is not critical because the Deep IV framework will
            # add extra layers on top for the mixture density network
            treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(d_z + d_x,)),
                                                keras.layers.Dropout(0.17),
                                                keras.layers.Dense(64, activation='relu'),
                                                keras.layers.Dropout(0.17),
                                                keras.layers.Dense(32, activation='relu'),
                                                keras.layers.Dropout(0.17)])

            # Define the response model neural network architecture
            # This will take the concatenation of one-dimensional values t and x as input,
            # so the input shape is (d_t + d_x,)
            # The output should match the shape of y, so it must have shape (d_y,) in this case
            # NOTE: For the response model, it is important to define the model *outside*
            #       of the lambda passed to the DeepIvEstimator, as we do here,
            #       so that the same weights will be reused in each instantiation
            response_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(1 + d_x,)),
                                               keras.layers.Dropout(0.17),
                                               keras.layers.Dense(64, activation='relu'),
                                               keras.layers.Dropout(0.17),
                                               keras.layers.Dense(32, activation='relu'),
                                               keras.layers.Dropout(0.17),
                                               keras.layers.Dense(1)])

            deepIv = DeepIV(n_components=10,  # number of gaussians in our mixture density network
                            m=lambda z, x: treatment_model(
                                keras.layers.concatenate([z, x])),  # treatment model
                            h=lambda t, x: response_model(keras.layers.concatenate([t, x])),  # response model
                            n_samples=1,  # number of samples to use to estimate the response
                            use_upper_bound_loss=False,  # whether to use an approximation to the true loss
                            # number of samples to use in second estimate of the response
                            # (to make loss estimate unbiased)
                            n_gradient_samples=1,
                            # Keras optimizer to use for training - see https://keras.io/optimizers/
                            optimizer='adam',
                            first_stage_options=fit_opts,
                            second_stage_options=fit_opts)

            deepIv.fit(Y=y, T=t, X=x, Z=z)
            # do something with predictions...
            deepIv.predict(T=t, X=x)
            assert (deepIv.effect(x).shape == (n,))

    # Doesn't work with CNTK backend as of 2018-07-17 - see https://github.com/keras-team/keras/issues/10715

    @pytest.mark.slow
    def test_deepiv_models(self):
        n = 2000
        epochs = 2
        e = np.random.uniform(low=-0.5, high=0.5, size=(n, 1))
        z = np.random.uniform(size=(n, 1))
        x = np.random.uniform(size=(n, 1)) + e
        p = x + z * e + np.random.uniform(size=(n, 1))
        y = p * x + e

        losses = []
        marg_effs = []

        z_fresh = np.random.uniform(size=(n, 1))
        e_fresh = np.random.uniform(low=-0.5, high=0.5, size=(n, 1))
        x_fresh = np.random.uniform(size=(n, 1)) + e_fresh
        p_fresh = x_fresh + z_fresh * e_fresh + np.random.uniform(size=(n, 1))
        y_fresh = p_fresh * x_fresh + e_fresh

        for (n1, u, n2) in [(2, False, None), (2, True, None), (1, False, 1)]:
            treatment_model = keras.Sequential([keras.layers.Dense(10, activation='relu', input_shape=(2,)),
                                                keras.layers.Dense(10, activation='relu'),
                                                keras.layers.Dense(10, activation='relu')])

            hmodel = keras.Sequential([keras.layers.Dense(10, activation='relu', input_shape=(2,)),
                                       keras.layers.Dense(10, activation='relu'),
                                       keras.layers.Dense(1)])

            deepIv = DeepIV(n_components=10,
                            m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])),
                            h=lambda t, x: hmodel(keras.layers.concatenate([t, x])),
                            n_samples=n1, use_upper_bound_loss=u, n_gradient_samples=n2,
                            first_stage_options={'epochs': epochs}, second_stage_options={'epochs': epochs})
            deepIv.fit(y, p, X=x, Z=z)

            losses.append(np.mean(np.square(y_fresh - deepIv.predict(p_fresh, x_fresh))))
            marg_effs.append(deepIv.marginal_effect(np.array([[0.3], [0.5], [0.7]]), np.array([[0.4], [0.6], [0.2]])))
        print("losses: {}".format(losses))
        print("marg_effs: {}".format(marg_effs))

    @pytest.mark.slow
    def test_deepiv_models_paper(self):
        def monte_carlo_error(g_hat, data_fn, ntest=5000, has_latent=False, debug=False):
            seed = np.random.randint(1e9)
            try:
                # test = True ensures we draw test set images
                x, z, t, y, g_true = data_fn(ntest, seed, test=True)
            except ValueError:
                warnings.warn("Too few images, reducing test set size")
                ntest = int(ntest * 0.7)
                # test = True ensures we draw test set images
                x, z, t, y, g_true = data_fn(ntest, seed, test=True)

            # re-draw to get new independent treatment and implied response
            t = np.linspace(np.percentile(t, 2.5), np.percentile(t, 97.5), ntest).reshape(-1, 1)
            # we need to make sure z _never_ does anything in these g functions (fitted and true)
            # above is necesary so that reduced form doesn't win
            if has_latent:
                x_latent, _, _, _, _ = data_fn(ntest, seed, images=False)
                y = g_true(x_latent, z, t)
            else:
                y = g_true(x, z, t)
            y_true = y.flatten()
            y_hat = g_hat(x, z, t).flatten()
            return ((y_hat - y_true)**2).mean()

        def one_hot(col, **kwargs):
            z = col.reshape(-1, 1)
            enc = OneHotEncoder(sparse=False, **kwargs)
            return enc.fit_transform(z)

        def sensf(x):
            return 2.0 * ((x - 5)**4 / 600 + np.exp(-((x - 5) / 0.5)**2) + x / 10. - 2)

        def emocoef(emo):
            emoc = (emo * np.array([1., 2., 3., 4., 5., 6., 7.])[None, :]).sum(axis=1)
            return emoc

        psd = 3.7
        pmu = 17.779
        ysd = 158.  # 292.
        ymu = -292.1

        def storeg(x, price):
            emoc = emocoef(x[:, 1:])
            time = x[:, 0]
            g = sensf(time) * emoc * 10. + (emoc * sensf(time) - 2.0) * (psd * price.flatten() + pmu)
            y = (g - ymu) / ysd
            return y.reshape(-1, 1)

        def demand(n, seed=1, ynoise=1., pnoise=1., ypcor=0.8, use_images=False, test=False):
            rng = np.random.RandomState(seed)

            # covariates: time and emotion
            time = rng.rand(n) * 10
            emotion_id = rng.randint(0, 7, size=n)
            emotion = one_hot(emotion_id, categories=[np.arange(7)])
            if use_images:
                idx = np.argsort(emotion_id)
                emotion_feature = np.zeros((0, 28 * 28))
                for i in range(7):
                    img = get_images(i, np.sum(emotion_id == i), seed, test)
                    emotion_feature = np.vstack([emotion_feature, img])
                reorder = np.argsort(idx)
                emotion_feature = emotion_feature[reorder, :]
            else:
                emotion_feature = emotion

            # random instrument
            z = rng.randn(n)

            # z -> price
            v = rng.randn(n) * pnoise
            price = sensf(time) * (z + 3) + 25.
            price = price + v
            price = (price - pmu) / psd

            # true observable demand function
            x = np.concatenate([time.reshape((-1, 1)), emotion_feature], axis=1)
            x_latent = np.concatenate([time.reshape((-1, 1)), emotion], axis=1)

            def g(x, z, p):
                return storeg(x, p)  # doesn't use z

            # errors
            e = (ypcor * ynoise / pnoise) * v + rng.randn(n) * ynoise * np.sqrt(1 - ypcor**2)
            e = e.reshape(-1, 1)

            # response
            y = g(x_latent, None, price) + e

            return (x,
                    z.reshape((-1, 1)),
                    price.reshape((-1, 1)),
                    y.reshape((-1, 1)),
                    g)

        def datafunction(n, s, images=False, test=False):
            return demand(n=n, seed=s, ypcor=0.5, use_images=images, test=test)

        n = 1000
        epochs = 50
        x, z, t, y, g_true = datafunction(n, 1)

        print("Data shapes:\n\
Features:{x},\n\
Instruments:{z},\n\
Treament:{t},\n\
Response:{y}".format(**{'x': x.shape, 'z': z.shape,
                        't': t.shape, 'y': y.shape}))

        losses = []

        for (n1, u, n2) in [(2, False, None), (2, True, None), (1, False, 1)]:
            treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(9,)),
                                                keras.layers.Dropout(0.17),
                                                keras.layers.Dense(64, activation='relu'),
                                                keras.layers.Dropout(0.17),
                                                keras.layers.Dense(32, activation='relu'),
                                                keras.layers.Dropout(0.17)])

            hmodel = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(9,)),
                                       keras.layers.Dropout(0.17),
                                       keras.layers.Dense(64, activation='relu'),
                                       keras.layers.Dropout(0.17),
                                       keras.layers.Dense(32, activation='relu'),
                                       keras.layers.Dropout(0.17),
                                       keras.layers.Dense(1)])

            deepIv = DeepIV(n_components=10,
                            m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])),
                            h=lambda t, x: hmodel(keras.layers.concatenate([t, x])),
                            n_samples=n1, use_upper_bound_loss=u, n_gradient_samples=n2,
                            first_stage_options={'epochs': epochs}, second_stage_options={'epochs': epochs})
            deepIv.fit(y, t, X=x, Z=z)

            losses.append(monte_carlo_error(lambda x, z, t: deepIv.predict(
                t, x), datafunction, has_latent=False, debug=False))
        print("losses: {}".format(losses))

    @pytest.mark.slow
    def test_deepiv_models_paper2(self):
        def monte_carlo_error(g_hat, data_fn, ntest=5000, has_latent=False, debug=False):
            seed = np.random.randint(1e9)
            try:
                # test = True ensures we draw test set images
                x, z, t, y, g_true = data_fn(ntest, seed, test=True)
            except ValueError:
                warnings.warn("Too few images, reducing test set size")
                ntest = int(ntest * 0.7)
                # test = True ensures we draw test set images
                x, z, t, y, g_true = data_fn(ntest, seed, test=True)

            # re-draw to get new independent treatment and implied response
            t = np.linspace(np.percentile(t, 2.5), np.percentile(t, 97.5), ntest).reshape(-1, 1)
            # we need to make sure z _never_ does anything in these g functions (fitted and true)
            # above is necesary so that reduced form doesn't win
            if has_latent:
                x_latent, _, _, _, _ = data_fn(ntest, seed, images=False)
                y = g_true(x_latent, z, t)
            else:
                y = g_true(x, z, t)
            y_true = y.flatten()
            y_hat = g_hat(x, z, t).flatten()
            return ((y_hat - y_true)**2).mean()

        def one_hot(col, **kwargs):
            z = col.reshape(-1, 1)
            enc = OneHotEncoder(sparse=False, **kwargs)
            return enc.fit_transform(z)

        def sensf(x):
            return 2.0 * ((x - 5)**4 / 600 + np.exp(-((x - 5) / 0.5)**2) + x / 10. - 2)

        def emocoef(emo):
            emoc = (emo * np.array([1., 2., 3., 4., 5., 6., 7.])[None, :]).sum(axis=1)
            return emoc

        psd = 3.7
        pmu = 17.779
        ysd = 158.  # 292.
        ymu = -292.1

        def storeg(x, price):
            emoc = emocoef(x[:, 1:])
            time = x[:, 0]
            g = sensf(time) * emoc * 10. + (6 * emoc * sensf(time) - 2.0) * (psd * price.flatten() + pmu)
            y = (g - ymu) / ysd
            return y.reshape(-1, 1)

        def demand(n, seed=1, ynoise=1., pnoise=1., ypcor=0.8, use_images=False, test=False):
            rng = np.random.RandomState(seed)

            # covariates: time and emotion
            time = rng.rand(n) * 10
            emotion_id = rng.randint(0, 7, size=n)
            emotion = one_hot(emotion_id, categories=[np.arange(7)])
            emotion_feature = emotion

            # random instrument
            z = rng.randn(n)

            # z -> price
            v = rng.randn(n) * pnoise
            price = sensf(time) * (z + 3) + 25.
            price = price + v
            price = (price - pmu) / psd

            # true observable demand function
            x = np.concatenate([time.reshape((-1, 1)), emotion_feature], axis=1)
            x_latent = np.concatenate([time.reshape((-1, 1)), emotion], axis=1)

            def g(x, z, p):
                return storeg(x, p)  # doesn't use z

            # errors
            e = (ypcor * ynoise / pnoise) * v + rng.randn(n) * ynoise * np.sqrt(1 - ypcor**2)
            e = e.reshape(-1, 1)

            # response
            y = g(x_latent, None, price) + e

            return (x,
                    z.reshape((-1, 1)),
                    price.reshape((-1, 1)),
                    y.reshape((-1, 1)),
                    g)

        def datafunction(n, s, images=False, test=False):
            return demand(n=n, seed=s, ypcor=0.5, use_images=images, test=test)

        n = 1000
        epochs = 20

        x, z, t, y, g_true = datafunction(n, 1)

        print("Data shapes:\n\
                Features:{x},\n\
                Instruments:{z},\n\
                Treament:{t},\n\
                Response:{y}".format(**{'x': x.shape, 'z': z.shape,
                                        't': t.shape, 'y': y.shape}))

        losses = []

        for (n1, u, n2) in [(2, False, None), (2, True, None), (1, False, 1)]:
            treatment_model = keras.Sequential([keras.layers.Dense(50, activation='relu', input_shape=(9,)),
                                                keras.layers.Dense(25, activation='relu'),
                                                keras.layers.Dense(25, activation='relu')])

            hmodel = keras.Sequential([keras.layers.Dense(50, activation='relu', input_shape=(9,)),
                                       keras.layers.Dense(25, activation='relu'),
                                       keras.layers.Dense(25, activation='relu'),
                                       keras.layers.Dense(1)])

            deepIv = DeepIV(n_components=10,
                            m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])),
                            h=lambda t, x: hmodel(keras.layers.concatenate([t, x])),
                            n_samples=n1, use_upper_bound_loss=u, n_gradient_samples=n2,
                            first_stage_options={'epochs': epochs}, second_stage_options={'epochs': epochs})
            deepIv.fit(y, t, X=x, Z=z)

            losses.append(monte_carlo_error(lambda x, z, t: deepIv.predict(
                t, x), datafunction, has_latent=False, debug=False))
        print("losses: {}".format(losses))

    @pytest.mark.slow
    def test_mog_models(self):
        d = 2
        n = 5

        theta = np.random.uniform(low=0.0, high=2 * np.pi, size=(5000, d))
        x = 10 * np.cos(theta) + np.random.normal(size=(5000, d))
        t = 10 * np.sin(theta) + np.random.normal(size=(5000, d))

        x_input = keras.layers.Input(shape=(d,))
        l1 = keras.layers.Dense(10, activation='relu')
        l2 = keras.layers.Dense(10, activation='relu')
        l3 = keras.layers.Dense(10, activation='relu')

        def norm(lr):
            return lr  # keras.layers.BatchNormalization()

        x_network = l3(norm(l2(norm(l1(x_input)))))

        t_input = keras.layers.Input(shape=(d,))

        pi, mu, sig = mog_model(n, 10, d)(x_network)
        ll = mog_loss_model(n, d)([pi, mu, sig, t_input])
        samp = mog_sample_model(n, d)

        samp2 = keras.layers.Concatenate()([samp([pi, mu, sig]), samp([pi, mu, sig])])

        # pi,mu,sig = MixtureOfGaussians(n, d)(x_network)
        # ll = MixtureOfGaussiansLogLoss(n, d)([pi,mu,sig,t_input])
        model = keras.engine.Model([x_input, t_input], [ll])
        model.add_loss(K.mean(ll))
        model.compile('nadam')
        model.fit([x, t], [], epochs=5)

        # For some reason this doesn't work at all when run against the CNTK backend...
        # model.compile('nadam', loss=lambda _,l:l)
        # model.fit([x,t], [np.zeros((5000,1))], epochs=500)

        model2 = keras.engine.Model([x_input], [pi, mu, sig])
        model3 = keras.engine.Model([x_input], [samp([pi, mu, sig])])
        model4 = keras.engine.Model([x_input], [samp2])

        print("samp2: {}".format(model4.predict(np.array([[0., 0.]]))))

        for x_i in [-10, -5, 0, 5, 10]:
            t = np.array([[np.sqrt(100 - x_i**2), -np.sqrt(100 - x_i**2)]])
            outs = model2.predict([np.array([[x_i, x_i]])])
            print(x_i, outs)

        # generate a valiation set
        x = 10 * np.cos(theta) + np.random.normal(size=(5000, d))
        t = 10 * np.sin(theta) + np.random.normal(size=(5000, d))
        pi, mu, sig = model2.predict([x])
        sampled_t = model3.predict([x])
        llm = model.predict([x, t])

        pi_o = np.tile([[0.25, 0.25, 0.25, 0.25, 0]], (5000, 1))
        x2 = np.sqrt(np.maximum(0, 100 - x**2)).reshape(-1, 1, 2)
        arrs = [np.array([f1, f2]) for f1 in [-1, 1] for f2 in [-1, 1]] + [np.zeros(2)]
        mu_o = np.concatenate([x2 * arr for arr in arrs], axis=1)
        sig_o = np.ones((5000, 5))

        print(pi[0], mu[0], sig[0], x[0], t[0])
        import io
        with io.open("sampled_{}.csv".format(K.backend()), 'w') as f:
            for (x1, x2), (t1, t2) in zip(x, sampled_t):
                f.write("{},{},{},{}\n".format(x1, t1, x2, t2))

    @pytest.mark.slow
    def test_mog_models2(self):
        def sample(n):
            x = np.random.uniform(size=2)
            return (n + 1) * x[0] if x[0] ** n > x[1] else sample(n)

        n_comp = 20

        x = np.random.uniform(high=2, size=2000)
        t = np.array([sample(n) for n in x])

        x_network = keras.Sequential([keras.layers.Dense(10, activation='relu'),
                                      keras.layers.Dense(10, activation='relu'),
                                      keras.layers.Dense(10, activation='relu')])

        x_input, t_input = [keras.layers.Input(shape=(d,)) for d in [1, 1]]

        pi, mu, sig = mog_model(n_comp, 10, 1)(x_network(x_input))
        ll = mog_loss_model(n_comp, 1)([pi, mu, sig, t_input])

        model = keras.engine.Model([x_input, t_input], [ll])
        model.add_loss(K.mean(ll))
        model.compile('nadam')
        model.fit([x, t], [], epochs=100)

        model2 = keras.engine.Model([x_input], [pi, mu, sig])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for x in [0, 1, 2]:
            pi, mu, sig = model2.predict(np.array([[x]]))
            mu = mu.reshape(-1)

            def f(t):
                return np.sum(pi / (np.sqrt(2 * np.pi) * sig) * np.exp(-np.square((t - mu) / sig) / 2))
            ts = np.linspace(-0.1, x + 1.1, 100)
            plt.figure()
            plt.plot(ts, [f(t) for t in ts])
            plt.plot(ts, [(t / (x + 1)) ** x for t in ts])
            plt.show()

    @pytest.mark.slow
    def test_deepiv_arbitrary_covariance(self):
        d = 5
        n = 5000
        # to generate a random symmetric positive semidefinite covariance matrix, we can use A*A^T
        A1 = np.random.normal(size=(d, d))
        cov1 = np.matmul(A1, np.transpose(A1))
        # convex combinations of semidefinite covariance matrices are themselves semidefinite
        A2 = np.random.normal(size=(d, d))
        cov2 = np.matmul(A2, np.transpose(A2))
        m1 = np.random.normal(size=(d,))
        m2 = np.random.normal(size=(d,))
        x = np.random.uniform(size=(n, 1))
        z = np.random.uniform(size=(n, 1))
        alpha = (x * x + z * z) / 2  # in range [0,1]
        t = np.array([np.random.multivariate_normal(m1 + alpha[i] * (m2 - m1),
                                                    cov1 + alpha[i] * (cov2 - cov1)) for i in range(n)])
        y = np.expand_dims(np.einsum('nx,nx->n', t, t), -1) + x
        results = []
        s = 6
        for (n1, u, n2) in [(2, False, None), (2, True, None), (1, False, 1)]:
            treatment_model = keras.Sequential([keras.layers.Dense(90, activation='relu', input_shape=(2,)),
                                                keras.layers.Dropout(0.2),
                                                keras.layers.Dense(60, activation='relu'),
                                                keras.layers.Dropout(0.2),
                                                keras.layers.Dense(30, activation='relu')])

            hmodel = keras.Sequential([keras.layers.Dense(90, activation='relu', input_shape=(d + 1,)),
                                       keras.layers.Dropout(0.2),
                                       keras.layers.Dense(60, activation='relu'),
                                       keras.layers.Dropout(0.2),
                                       keras.layers.Dense(30, activation='relu'),
                                       keras.layers.Dropout(0.2),
                                       keras.layers.Dense(1)])

            deepIv = DeepIV(n_components=s,
                            m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])),
                            h=lambda t, x: hmodel(keras.layers.concatenate([t, x])),
                            n_samples=n1, use_upper_bound_loss=u, n_gradient_samples=n2,
                            first_stage_options={'epochs': 20}, second_stage_options={'epochs': 20})
            deepIv.fit(y[:n // 2], t[:n // 2], X=x[:n // 2], Z=z[:n // 2])

            results.append({'s': s, 'n1': n1, 'u': u, 'n2': n2,
                            'loss': np.mean(np.square(y[n // 2:] - deepIv.predict(t[n // 2:], x[n // 2:]))),
                            'marg': deepIv.marginal_effect(np.array([[0.5] * d]), np.array([[1.0]]))})
        print(results)
