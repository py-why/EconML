import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
plt.style.use('ggplot')
from mcpy.utils import filesafe
import mcpy.metrics
import itertools

# example-dgp-specific
def plot_sweep(plot_name, sweep_keys, sweep_params, sweep_metrics, sweep_true_params, config):
    pass

# example-dgp-specific
def plot_metrics(plot_name, experiment_results, metric_results, true_params, config):
    """
    Plots all metrics that are listed in the metrics section of a config file
    which are not single summary statistics. Thus, this plots metrics that are a
    function of the points X. These metrics are all based on Theta(X).
    """
    for dgp_name in config['dgps'].keys(): # just one right now
        for metric_name in config['metrics'].keys():
            if metric_name not in config['single_summary_metrics']:
                for method_name in config['methods'].keys():
                    x, y = metric_results[dgp_name][method_name][metric_name]
                    plt.plot(x, y, label=method_name)
                plt.xlabel("X_test")
                plt.ylabel(metric_name)
                plt.legend()
                plt.savefig(plot_name + "_" + metric_name)
                plt.show()

# example-dgp-specific
def plot_visualization(plot_name, experiment_results, metric_results, true_params, config):
    """
    Plots the results of each method for each dgp vs. the true effect.
    """
    X_test = []
    for dgp_name in config['dgps'].keys(): # just one right now
        for method_name in config['methods'].keys():
            X_test = experiment_results[dgp_name][method_name][0][0][0]
            pred = np.array([experiment_results[dgp_name][method_name][i][0][1] for i in range(len(experiment_results[dgp_name][method_name]))])
            mean = np.mean(pred, axis=0)
            plt.plot(X_test, mean, label=method_name)
            plt.xlabel("X_test")
            plt.ylabel("Treatment Effect")
            lb = np.array([experiment_results[dgp_name][method_name][i][1][0] for i in range(len(experiment_results[dgp_name][method_name]))])
            ub = np.array([experiment_results[dgp_name][method_name][i][1][1] for i in range(len(experiment_results[dgp_name][method_name]))])
            lb_ = np.min(lb, axis=0)
            ub_ = np.max(ub, axis=0)
            plt.fill_between(X_test.reshape(100,), lb_, ub_, alpha=0.25)

        true = true_params[dgp_name][0]
        plt.plot(X_test, true, label='true effect')
        plt.legend()
        plt.savefig(plot_name)
        plt.show()

# example-dgp-specific
def plot_violin(plot_name, experiment_results, metric_results, true_params, config):
    """
    Plots all metrics that are single summary statistics, for each method. These
    are single numbers produced for each method for each experiment. They are not a function
    of X.
    """
    for dgp_name, dgp_fn in metric_results.items():
        n_methods = len(list(dgp_fn.keys()))
        for metric_name in next(iter(dgp_fn.values())).keys():
            if metric_name in config['single_summary_metrics']:
                plt.figure(figsize=(1.5 * n_methods, 2.5))
                plt.violinplot([dgp_fn[method_name][metric_name] for method_name in dgp_fn.keys()], showmedians=True)
                plt.xticks(np.arange(1, n_methods + 1), list(dgp_fn.keys()))
                plt.ylabel(metric_name)
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.show()
