# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import numpy as np
from joblib import Parallel, delayed
import joblib
import argparse
import importlib
from itertools import product
import collections

from copy import deepcopy
from mcpy.utils import filesafe
from mcpy import plotting

def check_valid_config(config):
    """
    Performs a basic check of the config file, checking if the necessary
    subsections are present.

    If multiple config files are being made that use the same dgps and/or methods,
    it may be helpful to tailor the config check to those dgps and methods. That way,
    one can check that the correct parameters are being provided for those dgps and methods.
    This is specific to one's implementation, however.
    """
    assert 'type' in config, "config dict must specify config type"
    assert 'dgps' in config, "config dict must contain dgps"
    assert 'dgp_opts' in config, "config dict must contain dgp_opts"
    assert 'method_opts' in config, "config dict must contain method_opts"
    assert 'mc_opts' in config, "config dict must contain mc_opts"
    assert 'metrics' in config, "config dict must contain metrics"
    assert 'methods' in config, "config dict must contain methods"
    assert 'plots' in config, "config dict must contain plots"
    assert 'single_summary_metrics' in config, "config dict must specify which metrics are plotted in a y-x plot vs. as a single value per dgp and method"
    assert 'target_dir' in config, "config must contain target_dir"
    assert 'reload_results' in config, "config must contain reload_results"
    assert 'n_experiments' in config['mc_opts'], "config[mc_opts] must contain n_experiments"
    assert 'seed' in config['mc_opts'], "config[mc_opts] must contain seed"

class MonteCarlo:
    """
    This class contains methods to run (multiple) monte carlo experiments

    Experiments are constructed from a config file, which mainly consists of
    references to the implementations of four different kinds of items, in
    addition to various parameters for the experiment. See the README for
    a descriptoin of the config file, or look at an example in the configs directory.
    The four main items are:
     - data generating processes (dgps): functions that generate data according to
     some assumed underlying model
     - methods: functions that take in data and produce other data. In our case,
     they train on data produced by DGPs and then produce counterfactual estimates
     - metrics: functions that take in the results of estimators and calculate metrics
     - plots: functions that take in the metric results, etc. and generate plots
    """

    def __init__(self, config):
        self.config = config
        check_valid_config(self.config)
        # these param strings are for properly naming results saved to disk
        config['param_str'] = '_'.join(['{}_{}'.format(filesafe(k), v) for k,v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(k), v) for k,v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(k), v) for k,v in self.config['method_opts'].items()])

    def experiment(self, instance_params, seed):
        """
        Given instance parameters to pass on to the data generating processes,
        runs an experiment on a single randomly generated instance of data and returns the
        parameter estimates for each method and the evaluated metrics for each method.

        Parameters
        ----------
        instance_params : dictionary
            instance paramaters that DGP functions may use
        seed : int
            random seed for random data generation

        Returns
        -------
        experiment_results : dictionary
            results of the experiment, depending on what the methods return.
            These are stored by dgp_name and then by method_name.
        true_params : dictionary
            true parameters of the DGP, indexed by dgp_name, used for metrics
            calculation downstream
        """
        np.random.seed(seed)

        experiment_results = {}
        true_params = {}
        for dgp_name, dgp_fn in self.config['dgps'].items():
            data, true_param = dgp_fn(self.config['dgp_opts'][dgp_name], instance_params[dgp_name], seed)
            true_params[dgp_name] = true_param
            experiment_results[dgp_name] = {}

            for method_name, method in self.config['methods'].items():
                experiment_results[dgp_name][method_name] = method(data, self.config['method_opts'][method_name], seed)

        return experiment_results, true_params

    def run(self):
        """
        Runs multiple experiments in parallel on randomly generated instances and samples and returns
        the results for each method and the evaluated metrics for each method across all
        experiments.

        Returns
        -------
        simulation_results : dictionary
            dictionary indexed by [dgp_name][method_name] for individual experiment results
        metric_results : dictionary
            dictionary indexed by [dgp_name][method_name][metric_name]
        true_param : dictinoary
            dictionary indexed by [dgp_name]
        """
        random_seed = self.config['mc_opts']['seed']

        if not os.path.exists(self.config['target_dir']):
            os.makedirs(self.config['target_dir'])

        instance_params = {}
        for dgp_name in self.config['dgps']:
            instance_params[dgp_name] = self.config['dgp_instance_fns'][dgp_name](self.config['dgp_opts'][dgp_name], random_seed)

        # results_file = os.path.join(self.config['target_dir'], 'results_{}.jbl'.format(self.config['param_str']))
        results_file = os.path.join(self.config['target_dir'], 'results_seed{}.jbl'.format(random_seed))
        if self.config['reload_results'] and os.path.exists(results_file):
            results = joblib.load(results_file)
        else:
            results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(self.experiment)(instance_params, random_seed + exp_id)
                    for exp_id in range(self.config['mc_opts']['n_experiments']))
            joblib.dump(results, results_file)

        simulation_results = {} # note that simulation_results is a vector of individual experiment_results. from experiment()
        metric_results = {}
        true_params = {}
        for dgp_name in self.config['dgps'].keys():
            simulation_results[dgp_name] = {}
            metric_results[dgp_name] = {}
            for method_name in self.config['methods'].keys():
                simulation_results[dgp_name][method_name] = [results[i][0][dgp_name][method_name] for i in range(self.config['mc_opts']['n_experiments'])]
                true_params[dgp_name] = [results[i][1][dgp_name] for i in range(self.config['mc_opts']['n_experiments'])]
                metric_results[dgp_name][method_name] = {}
                for metric_name, metric_fn in self.config['metrics'].items():
                # for metric_name, metric_fn in self.config['metrics'][method_name].items(): # for method specific parameters
                    metric_results[dgp_name][method_name][metric_name] = metric_fn(simulation_results[dgp_name][method_name], true_params[dgp_name])

        for plot_name, plot_fn in self.config['plots'].items():
        # for plot_name, plot_fn in self.config['plots'][method_name].items(): # for method specific plots
            if isinstance(plot_fn, dict):
                plotting.instance_plot(plot_name, simulation_results, metric_results, self.config, plot_fn)
            else:
                plot_fn(plot_name, simulation_results, metric_results, true_params, self.config)

        return simulation_results, metric_results, true_params

class MonteCarloSweep:
    """
    This class contains methods to run sets of multiple monte carlo experiments
    where each set of experiments has different parameters (for the dgps and methods, etc.).
    This enables sweeping through parameter values to generate results for each permutation
    of parameters. For example, running a simulation when the number of samples a specific DGP
    generates is 100, 1000, or 10000.
    """

    def __init__(self, config):
        self.config = config
        check_valid_config(self.config)
        config['param_str'] = '_'.join(['{}_{}'.format(filesafe(k), self.stringify_param(v)) for k,v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(k), self.stringify_param(v)) for k,v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(k), self.stringify_param(v)) for k,v in self.config['method_opts'].items()])

    def stringify_param(self, param):
        """
        Parameters
        ----------
        param : list
            list denoting the various values a parameter should take

        Returns
        -------
        A string representation of the range of the values that parameter will take
        """
        if hasattr(param, "__len__"):
            return '{}_to_{}'.format(np.min(param), np.max(param))
        else:
            return param

    def run(self):
        """
        Runs many monte carlo simulations for all the permutations of parameters
        specified in the config file.

        Returns
        -------
        sweep_keys : list
            list of all the permutations of parameters for each dgp
        sweep_sim_results : list
            list of simulation results for each permutation of parameters for each dgp
        sweep_metrics : list
            list of metric results for each permutation of parameters for each dgp
        sweep_true_params : list
            list of true parameters for each permutation of parameters for each dgp 
        """
        # currently duplicates computation for the dgps because all only one dgp param changed each config
        # need to make it so that every inst_config is different for each dgp
        for dgp_name in self.config['dgp_opts'].keys():
            dgp_sweep_params = []
            dgp_sweep_param_vals = []
            for dgp_key, dgp_val  in self.config['dgp_opts'][dgp_name].items():
                if hasattr(dgp_val, "__len__"):
                    dgp_sweep_params.append(dgp_key)
                    dgp_sweep_param_vals.append(dgp_val)
            sweep_keys = []
            sweep_sim_results = []
            sweep_metrics = []
            sweep_true_params = []
            inst_config = deepcopy(self.config)
            for vec in product(*dgp_sweep_param_vals):
                setting = list(zip(dgp_sweep_params, vec))
                for k,v in setting:
                    inst_config['dgp_opts'][dgp_name][k] = v
                simulation_results, metrics, true_params = MonteCarlo(inst_config).run()
                sweep_keys.append(setting)
                sweep_sim_results.append(simulation_results)
                sweep_metrics.append(metrics)
                sweep_true_params.append(true_params)

        for plot_name, plot_fn in self.config['sweep_plots'].items():
            if isinstance(plot_fn, dict):
                plotting.sweep_plot(plot_key, sweep_keys, sweep_sim_results, sweep_metrics, self.config, plot_fn)
            else:
                plot_fn(plot_name, sweep_keys, sweep_sim_results, sweep_metrics, sweep_true_params, self.config)

        return sweep_keys, sweep_sim_results, sweep_metrics, sweep_true_params
