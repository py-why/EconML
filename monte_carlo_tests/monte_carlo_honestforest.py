import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import warnings
import joblib
from econml.grf import RegressionForest


def monte_carlo():
    n = 5000
    d = 5
    x_grid = np.linspace(-1, 1, 1000)
    X_test = np.hstack([x_grid.reshape(-1, 1), np.random.normal(size=(1000, d - 1))])
    coverage = []
    exp_dict = {'point': [], 'low': [], 'up': []}
    for it in range(100):
        print(it)
        X = np.random.normal(0, 1, size=(n, d))
        y = X[:, 0] + np.random.normal(size=(n,))
        est = RegressionForest(n_estimators=1000, verbose=1)
        est.fit(X, y)
        point = est.predict(X_test)
        low, up = est.predict_interval(X_test, alpha=0.05)
        coverage.append((low <= x_grid) & (x_grid <= up))
        exp_dict['point'].append(point)
        exp_dict['low'].append(low)
        exp_dict['up'].append(up)

    if not os.path.exists('figures'):
        os.makedirs('figures')
    if not os.path.exists(os.path.join("figures", 'honestforest')):
        os.makedirs(os.path.join("figures", 'honestforest'))

    plt.figure()
    plt.plot(x_grid, np.mean(coverage, axis=0))
    plt.savefig('figures/honestforest/coverage.png')

    plt.figure()
    plt.plot(x_grid, np.sqrt(np.mean((np.array(exp_dict['point']) - x_grid)**2, axis=0)), label='RMSE')
    plt.savefig('figures/honestforest/rmse.png')

    plt.figure()
    plt.plot(x_grid, np.mean(np.array(exp_dict['up']) - np.array(exp_dict['low']), axis=0), label='length')
    plt.savefig('figures/honestforest/length.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monte Carlo Coverage Tests for the RegressionForest')
    monte_carlo()
