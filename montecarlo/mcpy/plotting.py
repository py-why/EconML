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

def plot_sweep(plot_name, sweep_keys, sweep_params, sweep_metrics, sweep_true_params, config):
    pass

def plot_metrics(plot_name, param_estimates, metric_results, true_params, config):
    # plt.figure(figsize=(10,6))
    for dgp_name in config['dgps'].keys(): # just one right now
        for metric_name in config['metrics'].keys():
            if metric_name not in config['per_plots']:
                for method_name in config['methods'].keys():
                    x, y = metric_results[dgp_name][method_name][metric_name]
                    plt.plot(x, y, label=method_name)
                plt.xlabel("X_test")
                plt.ylabel(metric_name)
                plt.legend()
                plt.show()
                # mean_rmse = np.mean(metric_results[dgp_name][method_name]['rmse'])
                # mean_conf_length = np.mean(metric_results[dgp_name][method_name]['conf_length'])
                # mean_coverage = np.mean(metric_results[dgp_name][method_name]['coverage'])
                # print("means for {}: rmse: {}, conf_length: {}, coverage: {}".format(method_name, mean_rmse, mean_conf_length, mean_coverage))

def plot_visualization(plot_name, param_estimates, metric_results, true_params, config):
    X_test = []
    for dgp_name in config['dgps'].keys(): # just one right now
        for method_name in config['methods'].keys():
            X_test = param_estimates[dgp_name][method_name][0][0][0]
            pred = np.array([param_estimates[dgp_name][method_name][i][0][1] for i in range(len(param_estimates[dgp_name][method_name]))])
            mean = np.mean(pred, axis=0)
            plt.plot(X_test, mean, label=method_name)
            plt.xlabel("X_test")
            plt.ylabel("Treatment Effect")
            lb = np.array([param_estimates[dgp_name][method_name][i][1][0] for i in range(len(param_estimates[dgp_name][method_name]))])
            ub = np.array([param_estimates[dgp_name][method_name][i][1][1] for i in range(len(param_estimates[dgp_name][method_name]))])
            lb_ = np.min(lb, axis=0)
            ub_ = np.max(ub, axis=0)
            plt.fill_between(X_test.reshape(100,), lb_, ub_, alpha=0.25)

        true = true_params[dgp_name][0]
        plt.plot(X_test, true, label='true effect')
        plt.legend()
        plt.show()

def plot_violin(plot_name, param_estimates, metric_results, true_params, config):
    for dgp_name, dgp_fn in metric_results.items():
        n_methods = len(list(dgp_fn.keys()))
        for metric_name in next(iter(dgp_fn.values())).keys():
            if metric_name in config['per_plots']:
                plt.figure(figsize=(1.5 * n_methods, 2.5))
                plt.violinplot([dgp_fn[method_name][metric_name] for method_name in dgp_fn.keys()], showmedians=True)
                plt.xticks(np.arange(1, n_methods + 1), list(dgp_fn.keys()))
                plt.ylabel(metric_name)
                plt.tight_layout()
                plt.show()

def plot_subset_param_histograms(param_estimates, metric_results, config, subset):
    for dgp_name, pdgp in param_estimates.items():
        n_methods = len(list(pdgp.keys()))
        n_params = config['dgp_opts']['kappa_gamma'] + 1
        plt.figure(figsize=(4 * n_params, 2 * n_methods))
        for it, m_name in enumerate(pdgp.keys()):
            for inner_it, i in enumerate(subset):
                plt.subplot(n_methods, n_params, it * n_params + inner_it + 1)
                plt.hist(pdgp[m_name][:, i])
                plt.title("{}[{}]. $\\mu$: {:.2f}, $\\sigma$: {:.2f}".format(m_name, i, np.mean(pdgp[m_name][:, i]), np.std(pdgp[m_name][:, i])))
        plt.tight_layout()
        plt.savefig(os.path.join(config['target_dir'], 'dist_dgp_{}_{}.png'.format(dgp_name, config['param_str'])), dpi=300)
        plt.close()
    return

def plot_param_histograms2(param_estimates, metric_results, config):
    for dgp_name, pdgp in param_estimates.items():
        n_methods = len(list(pdgp.keys()))
        n_params = next(iter(pdgp.values())).shape[1]
        plt.figure(figsize=(4 * n_params, 2 * n_methods))
        for it, m_name in enumerate(pdgp.keys()):
            for i in range(pdgp[m_name].shape[1]):
                plt.subplot(n_methods, n_params, it * n_params + i + 1)
                plt.hist(pdgp[m_name][:, i])
                plt.title("{}[{}]. $\\mu$: {:.2f}, $\\sigma$: {:.2f}".format(m_name, i, np.mean(pdgp[m_name][:, i]), np.std(pdgp[m_name][:, i])))
        plt.tight_layout()
        plt.savefig(os.path.join(config['target_dir'], 'dist_dgp_{}_{}.png'.format(dgp_name, config['param_str'])), dpi=300)
        plt.close()
    return

def plot_metrics2(param_estimates, metric_results, config):
    for dgp_name, mdgp in metric_results.items():
        n_methods = len(list(mdgp.keys()))
        for metric_name in next(iter(mdgp.values())).keys():
            plt.figure(figsize=(1.5 * n_methods, 2.5))
            plt.violinplot([mdgp[method_name][metric_name] for method_name in mdgp.keys()], showmedians=True)
            plt.xticks(np.arange(1, n_methods + 1), list(mdgp.keys()))
            plt.ylabel(metric_name)
            plt.tight_layout()
            plt.savefig(os.path.join(config['target_dir'], '{}_dgp_{}_{}.png'.format(filesafe(metric_name), dgp_name, config['param_str'])), dpi=300)
            plt.close()
    return

def plot_metric_comparisons(param_estimates, metric_results, config):
    for dgp_name, mdgp in metric_results.items():
        n_methods = len(list(mdgp.keys()))
        for metric_name in next(iter(mdgp.values())).keys():
            plt.figure(figsize=(1.5 * n_methods, 2.5))
            plt.violinplot([mdgp[method_name][metric_name] - mdgp[config['proposed_method']][metric_name] for method_name in mdgp.keys() if method_name != config['proposed_method']], showmedians=True)
            plt.xticks(np.arange(1, n_methods), [method_name for method_name in mdgp.keys() if method_name != config['proposed_method']])
            plt.ylabel('decrease in {}'.format(metric_name))
            plt.tight_layout()
            plt.savefig(os.path.join(config['target_dir'], '{}_decrease_dgp_{}_{}.png'.format(filesafe(metric_name), dgp_name, config['param_str'])), dpi=300)
            plt.close()
    return

def instance_plot(plot_name, param_estimates, metric_results, config, plot_config):
    methods = plot_config['methods'] if 'methods' in plot_config else list(config['methods'].keys())
    metrics = plot_config['metrics'] if 'metrics' in plot_config else list(config['metrics'].keys())
    dgps = plot_config['dgps'] if 'dgps' in plot_config else list(config['dgps'].keys())
    metric_transforms = plot_config['metric_transforms'] if 'metric_transforms' in plot_config else {'': mcpy.metrics.transform_identity}

    for tr_name, tr_fn in metric_transforms.items():
        for dgp_name in dgps:
            for metric_name in metrics:
                plt.figure(figsize=(1.5 * len(methods), 2.5))
                plt.violinplot([tr_fn(metric_results, dgp_name, method_name, metric_name, config) for method_name in methods], showmedians=True)
                plt.xticks(np.arange(1, len(methods) + 1), methods)
                plt.ylabel('{}({})'.format(tr_name, metric_name))
                plt.tight_layout()
                plt.savefig(os.path.join(config['target_dir'], '{}_{}_{}_dgp_{}_{}.png'.format(plot_name, filesafe(metric_name), tr_name, dgp_name, config['param_str'])), dpi=300)
                plt.close()
    return

def _select_config_keys(sweep_keys, select_vals, filter_vals):

    if select_vals is not None:
        mask_select = [all(any((p, v) in key for v in vlist) for p, vlist in select_vals.items()) for key in sweep_keys]
    else:
        mask_select = [True]*len(sweep_keys)
    if filter_vals is not None:
        mask_filter = [all(all((p, v) not in key for v in vlist) for p, vlist in filter_vals.items()) for key in sweep_keys]
    else :
        mask_filter = [True]*len(sweep_keys)
    mask = [ms and mf for ms, mf in zip(mask_select, mask_filter)]
    return mask

def sweep_plot_marginal_transformed_metric(transform_fn, transform_name, dgps, methods, metrics, plot_name, sweep_keys, sweep_params, sweep_metrics, config, param_subset={}, select_vals={}, filter_vals={}):

    sweeps = {}
    for dgp_key, dgp_val in config['dgp_opts'].items():
        if hasattr(dgp_val, "__len__"):
            sweeps[dgp_key] = dgp_val

    mask = _select_config_keys(sweep_keys, select_vals, filter_vals)
    if np.sum(mask) == 0:
        print("Filtering resulted in no valid configurations!")
        return

    for dgp in dgps:
        for metric in metrics:
            for param, param_vals in sweeps.items():
                if param_subset is not None and param not in param_subset:
                    continue
                plt.figure(figsize=(5, 3))
                for method in methods:
                    medians = []
                    mins = []
                    maxs = []
                    for val in param_vals:
                        subset = [transform_fn(metrics, dgp, method, metric, config) for key, metrics, ms
                                        in zip(sweep_keys, sweep_metrics, mask)
                                        if (param, val) in key and ms]
                        if len(subset) > 0:
                            grouped_results = np.concatenate(subset)
                            medians.append(np.median(grouped_results))
                            mins.append(np.min(grouped_results))
                            maxs.append(np.max(grouped_results))
                    plt.plot(param_vals, medians, label=method)
                    plt.fill_between(param_vals, maxs, mins, alpha=0.3)
                plt.legend()
                plt.xlabel(param)
                plt.ylabel('{}({})'.format(transform_name, metric))
                plt.tight_layout()
                plt.savefig(os.path.join(config['target_dir'], '{}_{}_{}_dgp_{}_growing_{}_{}.png'.format(plot_name, filesafe(metric), transform_name, dgp, filesafe(param), config['param_str'])), dpi=300)
                plt.close()

            for param1, param2 in itertools.combinations(sweeps.keys(), 2):
                if param_subset is not None and (param1, param2) not in param_subset and (param2, param1) not in param_subset:
                    continue
                x, y, z = [], [], []
                for method_it, method in enumerate(methods):
                    x.append([]), y.append([]), z.append([])
                    for val1, val2 in itertools.product(*[sweeps[param1], sweeps[param2]]):
                        subset = [transform_fn(metrics, dgp, method, metric, config) for key, metrics, ms
                                        in zip(sweep_keys, sweep_metrics, mask)
                                        if (param1, val1) in key and (param2, val2) in key and ms]
                        if len(subset) > 0:
                            grouped_results = np.concatenate(subset)
                            x[method_it].append(val1)
                            y[method_it].append(val2)
                            z[method_it].append(np.median(grouped_results))
                vmin = np.min(z)
                vmax = np.max(z)
                fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(4 * len(methods), 3))
                if not hasattr(axes, '__len__'):
                    axes = [axes]
                for method_it, (method, ax) in enumerate(zip(methods, axes)):
                    xi = np.linspace(np.min(x[method_it]), np.max(x[method_it]), 5*len(x[method_it]))
                    yi = np.linspace(np.min(y[method_it]), np.max(y[method_it]), 5*len(y[method_it]))
                    zi = griddata(np.array(x[method_it]), np.array(y[method_it]), np.array(z[method_it]), xi, yi, interp='linear')
                    ax.contour(xi, yi, zi, 15, linewidths=0.2, colors='k')
                    im = ax.pcolormesh(xi, yi, zi, cmap=plt.cm.Reds, vmin=vmin, vmax=vmax)
                    ax.contourf(xi, yi, zi, 15, cmap=plt.cm.Reds, vmin=vmin, vmax=vmax)
                    ax.scatter(np.array(x[method_it]), np.array(y[method_it]), alpha=0.5, s=3, c='b')
                    ax.set_xlabel(param1)
                    ax.set_ylabel(param2)
                    ax.set_title(method)

                plt.tight_layout()
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.72])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_ylabel('median {}({})'.format(transform_name, metric))
                plt.savefig(os.path.join(config['target_dir'], '{}_{}_{}_dgp_{}_growing_{}_and_{}_{}.png'.format(plot_name, filesafe(metric), transform_name, dgp, filesafe(param1), filesafe(param2), config['param_str'])), dpi=300)
                plt.close()

def sweep_plot(plot_name, sweep_keys, sweep_params, sweep_metrics, config, plot_config):
    param_subset = plot_config['varying_params'] if 'varying_params' in plot_config else None
    select_vals = plot_config['select_vals'] if 'select_vals' in plot_config else {}
    filter_vals = plot_config['filter_vals'] if 'filter_vals' in plot_config else {}
    methods = plot_config['methods'] if 'methods' in plot_config else list(config['methods'].keys())
    metrics = plot_config['metrics'] if 'metrics' in plot_config else list(config['metrics'].keys())
    dgps = plot_config['dgps'] if 'dgps' in plot_config else list(config['dgps'].keys())
    metric_transforms = plot_config['metric_transforms'] if 'metric_transforms' in plot_config else {'': mcpy.metrics.transform_identity}

    for tr_name, tr_fn in metric_transforms.items():
        sweep_plot_marginal_transformed_metric(tr_fn, tr_name, dgps, methods, metrics, plot_name, sweep_keys, sweep_params, sweep_metrics, config,
                                                param_subset=param_subset, select_vals=select_vals, filter_vals=filter_vals)
