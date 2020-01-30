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

def _check_valid_config(config):
    assert 'dgps' in config, "config dict must contain dgps"
    assert 'dgp_opts' in config, "config dict must contain dgp_opts"
    assert 'method_opts' in config, "config dict must contain method_opts"
    assert 'mc_opts' in config, "config dict must contain mc_opts"
    assert 'metrics' in config, "config dict must contain metrics"
    assert 'methods' in config, "config dict must contain methods"
    assert 'plots' in config, "config dict must contain plots"
    assert 'target_dir' in config, "config must contain target_dir"
    assert 'reload_results' in config, "config must contain reload_results"
    assert 'n_experiments' in config['mc_opts'], "config[mc_opts] must contain n_experiments"
    assert 'seed' in config['mc_opts'], "config[mc_opts] must contain seed"

class MonteCarlo:

    def __init__(self, config):
        self.config = config
        _check_valid_config(self.config)
        config['param_str'] = '_'.join(['{}_{}'.format(filesafe(k), v) for k,v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(k), v) for k,v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(k), v) for k,v in self.config['method_opts'].items()])
        return

    # TODO: add dgp-specific, method-specific parameters and metrics
    def experiment(self, instance_params, seed):
        ''' Runs an experiment on a single randomly generated instance and sample and returns
        the parameter estimates for each method and the evaluated metrics for each method
        '''
        np.random.seed(seed)

        param_estimates = {}
        true_params = {}
        for dgp_name, dgp_fn in self.config['dgps'].items():
            data, true_param = dgp_fn(self.config['dgp_opts'][dgp_name], instance_params[dgp_name], seed)
            true_params[dgp_name] = true_param
            param_estimates[dgp_name] = {}

            for method_name, method in self.config['methods'].items():
                param_estimates[dgp_name][method_name] = method(data, self.config['method_opts'][method_name], seed)

        return param_estimates, true_params

    def run(self):
        ''' Runs multiple experiments in parallel on randomly generated instances and samples and returns
        the parameter estimates for each method and the evaluated metrics for each method across all
        experiments
        '''
        random_seed = self.config['mc_opts']['seed']

        if not os.path.exists(self.config['target_dir']):
            os.makedirs(self.config['target_dir'])

        instance_params = {}
        for dgp_name in self.config['dgps']:
            instance_params[dgp_name] = self.config['dgp_instance_fns'][dgp_name](self.config['dgp_opts'][dgp_name], random_seed)

        # results_file = os.path.join(self.config['target_dir'], 'results_{}.jbl'.format(self.config['param_str']))
        # what exactly is reload results for?
        results_file = os.path.join(self.config['target_dir'], 'results_seed{}.jbl'.format(random_seed))
        if self.config['reload_results'] and os.path.exists(results_file):
            results = joblib.load(results_file)
        else:
            results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(self.experiment)(instance_params, random_seed + exp_id)
                    for exp_id in range(self.config['mc_opts']['n_experiments']))
            joblib.dump(results, results_file)

        # import pdb; pdb.set_trace()

        param_estimates = {}
        metric_results = {}
        true_params = {}
        for dgp_name in self.config['dgps'].keys():
            param_estimates[dgp_name] = {}
            metric_results[dgp_name] = {}
            for method_name in self.config['methods'].keys():
                # param_estimates[dgp_name][method_name] = np.array([results[i][0][dgp_name][method_name] for i in range(self.config['mc_opts']['n_experiments'])])
                param_estimates[dgp_name][method_name] = [results[i][0][dgp_name][method_name] for i in range(self.config['mc_opts']['n_experiments'])]
                true_params[dgp_name] = [results[i][1][dgp_name] for i in range(self.config['mc_opts']['n_experiments'])]
                # import pdb; pdb.set_trace()
                metric_results[dgp_name][method_name] = {}
                for metric_name, metric_fn in self.config['metrics'].items():
                # for metric_name, metric_fn in self.config['metrics'][method_name].items(): # for method specific parameters
                    param_estimate_inputs = [results[i][0][dgp_name][method_name] for i in range(self.config['mc_opts']['n_experiments'])]
                    true_param_inputs = [results[i][1][dgp_name] for i in range(self.config['mc_opts']['n_experiments'])]
                    # import pdb; pdb.set_trace()
                    metric_results[dgp_name][method_name][metric_name] = metric_fn(param_estimate_inputs, true_param_inputs)

        for plot_name, plot_fn in self.config['plots'].items():
        # for plot_name, plot_fn in self.config['plots'][method_name].items(): # for method specific plots
            if isinstance(plot_fn, dict):
                plotting.instance_plot(plot_name, param_estimates, metric_results, self.config, plot_fn)
            else:
                plot_fn(plot_name, param_estimates, metric_results, true_params, self.config)

        return param_estimates, metric_results, true_params

class MonteCarloSweep:

    def __init__(self, config):
        self.config = config
        _check_valid_config(self.config)
        config['param_str'] = '_'.join(['{}_{}'.format(filesafe(k), self._stringify_param(v)) for k,v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(k), self._stringify_param(v)) for k,v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(k), self._stringify_param(v)) for k,v in self.config['method_opts'].items()])
        return

    def _stringify_param(self, param):
        if hasattr(param, "__len__"):
            return '{}_to_{}'.format(np.min(param), np.max(param))
        else:
            return param

    def run(self):
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
            sweep_params = []
            sweep_metrics = []
            sweep_true_params = []
            inst_config = deepcopy(self.config)
            for vec in product(*dgp_sweep_param_vals):
                setting = list(zip(dgp_sweep_params, vec))
                for k,v in setting:
                    inst_config['dgp_opts'][dgp_name][k] = v
                params, metrics, true_params = MonteCarlo(inst_config).run()
                sweep_keys.append(setting)
                sweep_params.append(params)
                sweep_metrics.append(metrics)
                sweep_true_params.append(true_params)

        for plot_name, plot_fn in self.config['sweep_plots'].items():
            if isinstance(plot_fn, dict):
                plotting.sweep_plot(plot_key, sweep_keys, sweep_params, sweep_metrics, self.config, plot_fn)
            else:
                plot_fn(plot_name, sweep_keys, sweep_params, sweep_metrics, sweep_true_params, self.config)

        return sweep_keys, sweep_params, sweep_metrics, sweep_true_params

def monte_carlo_main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args(sys.argv[1:])

    config = importlib.import_module(args.config)
    MonteCarlo(config.CONFIG).run()

if __name__=="__main__":
    monte_carlo_main()
