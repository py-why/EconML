# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os.path
import pandas as pd
import numpy as np
import urllib.request
from econml.utilities import reshape, shape
from econml.dml import LinearDML
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
import pytest


@pytest.mark.xfail(
    reason="This test used to work, but fully downloading the blob has become flaky. Needs investigation.")
def test_dominicks():
    file_name = "oj_large.csv"
    if not os.path.isfile(file_name):
        print("Downloading file (this might take a few seconds)...")
        urllib.request.urlretrieve(
            "https://msalicedatapublic.blob.core.windows.net/datasets/OrangeJuice/oj_large.csv", file_name)
    oj_data = pd.read_csv(file_name)

    brands = sorted(set(oj_data["brand"]))
    stores = sorted(set(oj_data["store"]))

    featnames = ["week", "feat"] + list(oj_data.columns[6:])

    # Preprocess data
    import datetime
    import numpy as np

    # Convert 'week' to a date
    # week_zero = datetime.datetime.strptime("09/07/89", "%m/%d/%y")
    # oj_data["week"] = pd.to_timedelta(oj_data["week"], unit='w') + week_zero

    # Take log of price
    oj_data["logprice"] = np.log(oj_data["price"])
    oj_data.drop("price", axis=1, inplace=True)

    # Make brand numeric
    oj_data["brand"] = [brands.index(b) for b in oj_data["brand"]]

    class PriceFeaturizer(TransformerMixin):
        def __init__(self, n_prods, own_price=True,
                     cross_price_groups=False, cross_price_indiv=True, per_product_effects=True):
            base_arrays = []
            effect_names = []
            one_hots = [(0,) * p + (1,) + (0,) * (n_prods - p - 1) for p in range(n_prods)]
            if own_price:
                base_arrays.append(np.eye(n_prods))
                effect_names.append("own price")
            if cross_price_groups:
                base_arrays.append((np.ones((n_prods, n_prods)) - np.eye(n_prods)) / (n_prods - 1))
                effect_names.append("group cross price")
            if cross_price_indiv:
                for p in range(n_prods):
                    base_arrays.append(one_hots[p] * np.ones((n_prods, 1)) - np.diag(one_hots[p]))
                    effect_names.append("cross price effect {} ->".format(p))
            if per_product_effects:
                all = [(np.diag(one_hots[p]) @ arr, nm + " {}".format(p))
                       for arr, nm in zip(base_arrays, effect_names) for p in range(n_prods)]
                # remove meaningless features (e.g. cross-price effects of products on themselves),
                # which have all zero coeffs
                nonempty = [(arr, nm) for arr, nm in all if np.count_nonzero(arr) > 0]
                self._features = [arr for arr, _ in nonempty]
                self._names = [nm for _, nm in nonempty]
            else:
                self._features = base_arrays
                self._names = effect_names

        def fit(self, X):
            self._is_fitted = True
            assert shape(X)[1] == 0
            return self

        def transform(self, X):
            assert self._is_fitted
            assert shape(X)[1] == 0
            return np.tile(self._features, (shape(X)[0], 1, 1, 1))

        @property
        def names(self):
            return self._names

    for name, op, xp_g, xp_i, pp in [("Homogeneous treatment effect", True, False, False, False),
                                     ("Heterogeneous treatment effects", True, False, False, True),
                                     (("Heterogeneous treatment effects"
                                       " with group effects"), True, True, False, True),
                                     (("Heterogeneous treatment effects"
                                       " with cross price effects"), True, False, True, True)]:

        print(name)
        np.random.seed(42)

        ft = PriceFeaturizer(n_prods=3, own_price=op, cross_price_groups=xp_g,
                             cross_price_indiv=xp_i, per_product_effects=pp)
        names = ft.names
        dml = LinearDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        featurizer=ft,
                        cv=2)

        effects = []
        for store in stores:
            data = oj_data[oj_data['store'] == store].sort_values(by=['week', 'brand'])
            dml.fit(T=reshape(data.as_matrix(["logprice"]), (-1, 3)),
                    Y=reshape(data.as_matrix(["logmove"]), (-1, 3)),
                    W=reshape(data.as_matrix(featnames), (-1, 3 * len(featnames))))
            effects.append(dml.coef_)
        effects = np.array(effects)
        for nm, eff in zip(names, effects.T):
            print(" Effect: {}".format(nm))
            print("   Mean: {}".format(np.mean(eff)))
            print("   Std.: {}".format(np.std(eff)))

    class ConstFt(TransformerMixin):
        def fit(self, X):
            return self

        def transform(self, X):
            return np.ones((shape(X)[0], 1))

    print("Vanilla HTE+XP")

    np.random.seed(42)
    dml = LinearDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestRegressor(),
                    featurizer=ConstFt(),
                    cv=2)

    effects = []
    for store in stores:
        data = oj_data[oj_data['store'] == store].sort_values(by=['week', 'brand'])
        dml.fit(T=reshape(data.as_matrix(["logprice"]), (-1, 3)),
                Y=reshape(data.as_matrix(["logmove"]), (-1, 3)),
                W=reshape(data.as_matrix(featnames), (-1, 3 * len(featnames))))
        effects.append(dml.coef_)
    effects = np.array(effects)
    names = ["{} on {}".format(i, j) for j in range(3) for i in range(3)]
    for nm, eff in zip(names, reshape(effects, (-1, 9)).T):
        print(" Effect: {}".format(nm))
        print("   Mean: {}".format(np.mean(eff)))
        print("   Std.: {}".format(np.std(eff)))
