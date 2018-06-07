import argparse
import copy
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import time
from joblib import Parallel, delayed
from matplotlib import rcParams, cm, rc
from sklearn.metrics import r2_score

matplotlib.rcParams['font.family'] = "serif"

###################
# Global settings #
###################
# Global plotting controls
# Control for support size, can control for more
plot_controls = ["support"]
label_order = ["ORF-CV", "ORF", "GRF-xW", "GRF-x", "GRF-Res", "HeteroDML-Lasso", "HeteroDML-RF"]
corresponding_str = ["OrthoForestCV", "OrthoForest", "GRF_Wx", "GRF_x", 
                    "GRF_res_Wx", "HeteroDML", "ForestHeteroDML"]

##################
# File utilities #
##################
def has_plot_controls(fname, control_combination):
    for c in control_combination:
        if "_{0}_".format(c) not in fname:
            return False
    return True

def get_file_key(fname):
    if "GRF" in fname:
        return "_" + "_".join(re.split("GRF_", fname)[0].split("_")[1:])
    else:
        return "_" + "_".join(re.split("results", fname)[0].split("_")[1:])

def sort_fnames(file_names):
    sorted_file_names = []
    label_indices = []
    for i, s in enumerate(corresponding_str):
        for f in file_names:
            if ((f.split("_")[0]==s and "GRF" not in f) 
                or ("_{0}_".format(s) in f and "GRF" in f)):
                sorted_file_names.append(f)
                label_indices.append(i)
                break
    return sorted_file_names, np.array(label_order)[label_indices]
        
def get_file_groups(agg_fnames, plot_controls):
    all_file_names = {}
    control_values = []
    for control in plot_controls:
        vals = set()
        for fname in agg_fnames:
            control_prefix = control + '_'
            val = re.search(control_prefix + '(\d+)', fname).group(1)
            vals.add(control_prefix + val)
        control_values.append(list(vals))
    control_combinations = list(itertools.product(*control_values))
    for control_combination in control_combinations:
        file_names = [f for f in agg_fnames if has_plot_controls(f, control_combination)]
        file_key = get_file_key(file_names[0])
        all_file_names[file_key], final_labels = sort_fnames(file_names)
    return all_file_names, final_labels

def merge_results(sf, input_dir, output_dir, split_files_seeds):
    name_template = "{0}seed_{1}_{2}"
    seeds = split_files_seeds[sf]
    df = pd.read_csv(os.path.join(input_dir, name_template.format(sf[0], seeds[0], sf[1])))
    te_idx = len([c for c in df.columns if bool(re.search("TE_[0-9]", c))])
    for i, seed in enumerate(seeds[1:]):
        new_df = pd.read_csv(os.path.join(input_dir, name_template.format(sf[0], seed, sf[1])))
        te_cols = [c for c in new_df.columns if bool(re.search("TE_[0-9]", c))]
        for te_col in te_cols:
            df["TE_"+str(te_idx)] = new_df[te_col]
            te_idx += 1
    agg_fname = os.path.join(output_dir, sf[0]+sf[1])
    df.to_csv(agg_fname, index=False)

def get_results(fname, dir_name):
    df = pd.read_csv(os.path.join(dir_name, fname))
    return df[[c for c in df.columns if "x" in c]+[c for c in df.columns if "TE_" in c]]

def save_plots(fig, fname, lgd=None):
    jpg_low_res_path = os.path.join(output_dir, "jpg_low_res")
    if not os.path.exists(jpg_low_res_path):
        os.makedirs(jpg_low_res_path)
    jpg_high_res_path = os.path.join(output_dir, "jpg_high_res")
    if not os.path.exists(jpg_high_res_path):
        os.makedirs(jpg_high_res_path)
    pdf_low_res_path = os.path.join(output_dir, "pdf_low_res")
    if not os.path.exists(pdf_low_res_path):
        os.makedirs(pdf_low_res_path)
    if lgd is None:
        fig.savefig(os.path.join(jpg_low_res_path, "{0}.png".format(fname)), bbox_inches='tight')
        fig.savefig(os.path.join(jpg_high_res_path, "{0}.png".format(fname)), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(pdf_low_res_path, "{0}.pdf".format(fname)), bbox_inches='tight')
    else:
        fig.savefig(os.path.join(jpg_low_res_path, "{0}.png".format(fname)), bbox_inches='tight', bbox_extra_artists=(lgd,))
        fig.savefig(os.path.join(jpg_high_res_path, "{0}.png".format(fname)), dpi=300, bbox_inches='tight', bbox_extra_artists=(lgd,))
        fig.savefig(os.path.join(pdf_low_res_path, "{0}.pdf".format(fname)), bbox_inches='tight', bbox_extra_artists=(lgd,))

##################
# Plotting utils #
##################
def get_r2(df):
    r2_scores = np.array([r2_score(df["TE_hat"], df[c]) for c in df.columns if bool(re.search('TE_[0-9]+', c))])
    return r2_scores

def get_metrics(dfs):
    biases = np.zeros((len(dfs[0]), len(dfs)))
    variances = np.zeros((len(dfs[0]), len(dfs)))
    rmses = np.zeros((len(dfs[0]), len(dfs)))
    r2_scores = []
    for i, df in enumerate(dfs):
        # bias
        treatment_effects = df[[c for c in df.columns if bool(re.search('TE_[0-9]+', c))]]
        bias = np.abs(np.mean(treatment_effects, axis=1) - df["TE_hat"])
        biases[:, i] = np.abs(np.mean(treatment_effects, axis=1) - df["TE_hat"])
        # var
        variance = np.std(treatment_effects, axis=1)
        variances[:, i] = np.std(treatment_effects, axis=1)
        # rmse
        rmse = np.mean(((treatment_effects.T - df["TE_hat"].values).T)**2, axis=1)
        rmses[:, i] = np.mean(((treatment_effects.T - df["TE_hat"].values).T)**2, axis=1)
        # r2
        r2_scores.append(get_r2(df))
    bias_lims = {"std": np.std(biases, axis=0), "mean": np.mean(biases, axis=0)}
    var_lims = {"std": np.std(variances, axis=0), "mean": np.mean(variances, axis=0)}
    rmse_lims = {"std": np.std(rmses, axis=0), "mean": np.mean(rmses, axis=0)}
    print(r2_scores)
    r2_lims = {"std": [np.std(r2_scores[i]) for i in range(len(r2_scores))], "mean": [np.mean(r2_scores[i]) for i in range(len(r2_scores))]}
    return {"bias": bias_lims, "var": var_lims, "rmse": rmse_lims, "r2": r2_lims}

def generic_joint_plots(file_key, dfs, labels, file_name_prefix):
    m = min(4, len(dfs))
    n = np.ceil((len(dfs)) / m) 
    fig = plt.figure(figsize=(10, 5))
    ymax =  max([max(df["TE_hat"]) for df in dfs])+1
    print(file_key)
    print(len(dfs))
    print(labels)
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(n, m, i+1)
        treatment_effects = df[[c for c in df.columns if bool(re.search('TE_[0-9]+', c))]]
        y = np.mean(treatment_effects, axis=1)
        err_up = np.percentile(treatment_effects, 95, axis=1)
        err_bottom = np.percentile(treatment_effects, 5, axis=1)
        ax.fill_between(df["x0"], err_up, err_bottom, alpha=0.5)
        if i == 0:
            ax.plot(df["x0"], y, label='Mean estimate')
            ax.plot(df["x0"], df["TE_hat"].values, 'b--', label='True effect')
        else:
            ax.plot(df["x0"], y)
            ax.plot(df["x0"], df["TE_hat"].values, 'b--', label=None)
        if i%m==0:
            ax.set_ylabel("Treatment effect")
        ax.set_ylim(ymax=ymax)
        ax.set_title(labels[i])
        if i + 1 > m*(n-1):
            ax.set_xlabel("x")    
    fig.legend(loc=(0.8, 0.25))
    fig.tight_layout()
    save_plots(fig, file_name_prefix)
    plt.clf()

def metrics_subfig(dfs, ax, metric, c_scheme=0):
    if c_scheme == 0:
        palette = plt.get_cmap('Set1')
    else:
        palette = plt.get_cmap('tab20b')
    if metric == "bias":
        biases = np.zeros((len(dfs[0]), len(dfs)))
        for i, df in enumerate(dfs):
            treatment_effects = df[[c for c in df.columns if bool(re.search('TE_[0-9]+', c))]]
            bias = np.abs(np.mean(treatment_effects, axis=1) - df["TE_hat"])
            biases[:, i] = np.abs(np.mean(treatment_effects, axis=1) - df["TE_hat"])
        vparts = ax.violinplot(biases, showmedians=True)
        ax.set_title("Bias")
    elif metric=="variance":
        variances = np.zeros((len(dfs[0]), len(dfs)))
        for i, df in enumerate(dfs):
            treatment_effects = df[[c for c in df.columns if bool(re.search('TE_[0-9]+', c))]]
            variance = np.std(treatment_effects, axis=1)
            variances[:, i] = np.std(treatment_effects, axis=1)
        vparts = ax.violinplot(variances, showmedians=True)
        ax.set_title("Variance")
    elif metric=="rmse":
        rmses = np.zeros((len(dfs[0]), len(dfs)))
        for i, df in enumerate(dfs):
            treatment_effects = df[[c for c in df.columns if bool(re.search('TE_[0-9]+', c))]]
            rmse = np.mean(((treatment_effects.T - df["TE_hat"].values).T)**2, axis=1)
            rmses[:, i] = np.mean(((treatment_effects.T - df["TE_hat"].values).T)**2, axis=1)
        vparts = ax.violinplot(rmses, showmedians=True)
        ax.set_title("RMSE")
    elif metric == "R2":
        r2_scores = []
        for i, df in enumerate(dfs):
            r2_scores.append(get_r2(df))
        vparts = ax.violinplot(r2_scores, showmedians=True)
        ax.set_title("$R^2$")
    else:
        print("No such metric")
        return 0
    cs = [0, 3, 12, 14, 15, 4, 6]
    ax.set_xticks([])
    for i, pc in enumerate(vparts['bodies']):
        if i < 5:
            c = i
        else:
            c = i+1
        if c_scheme == 1:
            c = cs[i]
        pc.set_facecolor(palette(c))
        pc.set_edgecolor(palette(c))
        pc.set_alpha(0.9)
    
    alpha = 0.7
    vparts['cbars'].set_color('black')
    vparts['cbars'].set_alpha(0.3)
    vparts['cbars'].set_linestyle('--')
    
    vparts['cmins'].set_color('black')
    vparts['cmins'].set_alpha(alpha)
    
    vparts['cmaxes'].set_color('black')
    vparts['cmaxes'].set_alpha(alpha)
    
    vparts['cmedians'].set_color('black')
    vparts['cmedians'].set_alpha(alpha)
    return vparts['bodies']

def metrics_plots(file_key, dfs, labels, c_scheme, file_name_prefix):
    metrics = ["bias", "variance", "rmse"]
    m = 1
    n = len(metrics)
    fig = plt.figure(figsize=(12*n/3, 3))
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(m, n, i+1)
        vbodies = metrics_subfig(dfs, ax, metric, c_scheme)
    lgd = fig.legend(vbodies, labels, ncol=len(labels), loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    save_plots(fig, file_name_prefix, lgd)
    plt.clf()

def support_plots(all_metrics, labels, file_name_prefix):
    palette = plt.get_cmap('Set1')
    x = sorted(list(all_metrics.keys()))
    metrics = ["bias", "var", "rmse"]
    titles = ["Bias", "Variance", "RMSE"]
    m = 1
    n = len(metrics)
    fig = plt.figure(figsize=(12*n/3, 3))
    all_plots = []
    for it, metric in enumerate(metrics):
        ax = fig.add_subplot(m, n, it+1)
        for i, l in enumerate(labels):
            if i < 5:
                c = i
            else:
                c = i+1
            err = np.array([all_metrics[j][metric]["std"][i] for j in x])
            mid = np.array([all_metrics[j][metric]["mean"][i] for j in x])
            p = ax.fill_between(x, mid-err/6, mid+err/6, alpha=0.5, color=palette(c))
            ax.plot(x, mid, label=labels[i], color=palette(c))
            if it == 0:
                p1 = copy.copy(p)
                p1.set_alpha(1.0)
                all_plots.append(p1)
        ax.set_title(titles[it])
        ax.set_xlabel("Support size")
    fig.legend(all_plots, labels, ncol=len(labels), loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    save_plots(fig, file_name_prefix)
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directory for saving results", default=".")
    parser.add_argument("--input_dir", type=str, help="", default=".")
    parser.add_argument("-merge", action='store_true')
    
    args = parser.parse_args(sys.argv[1:])
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    all_files = os.listdir(input_dir)
    results_files = [f for f in all_files if f.endswith("results.csv") and "seed" in f]
    split_files = set([(re.split("seed_[0-9]+_", f)[0], re.split("seed_[0-9]+_", f)[1]) for f in results_files])
    split_files_seeds = {k:[int(re.search("seed_(\d+)_", f).group(1)) for f in results_files if f.startswith(k[0]) and f.endswith(k[1])] for k in split_files}
    name_template = "{0}seed_{1}_{2}"
    agg_fnames = [sf[0] + sf[1] for sf in split_files]
    if args.merge:
        Parallel(n_jobs=-1, verbose=3)(delayed(merge_results)(sf, input_dir, output_dir, split_files_seeds) for sf in split_files)
    
    agg_file_groups, labels = get_file_groups(agg_fnames, plot_controls)
    print(agg_fnames)
    print(agg_file_groups)
    all_metrics = {}
    metrics_by_xgroup = [{}, {}]
    for g in agg_file_groups:
        agg_file_group = agg_file_groups[g]
        dfs = [get_results(fname, output_dir) for fname in agg_file_group]
        all_metrics[int(re.search("support_" + '(\d+)', g).group(1))] = get_metrics(dfs)
        # Infer feature dimension
        n_x  = len([c for c in dfs[0].columns if bool(re.search("x[0-9]", c))])
        if n_x == 1:
            generic_joint_plots(g, dfs, labels, "{0}{1}".format("Example", g))
            metrics_plots(g, dfs, labels, 0, "{0}{1}".format("Metrics", g))
        else:
            metrics_plots(g, dfs, labels, 0, "{0}_x1={2}{1}".format("Metrics", g, "all"))
            for i in range(2):
                dfs1 = [df[df["x1"]==i] for df in dfs]
                generic_joint_plots(g, dfs1, labels, "{0}_x1={2}{1}".format("Example", g, str(i)))
                metrics_plots(g, dfs1, labels, 0, "{0}_x1={2}{1}".format("Metrics", g, str(i)))
                metrics_by_xgroup[i][int(re.search("support_" + '(\d+)', g).group(1))] = get_metrics(dfs1)
    # Metrics by support plots
    if n_x == 1:
        support_plots(all_metrics, labels, "{0}".format("Metrics_by_support"))
    else:
        support_plots(all_metrics, labels, "{0}_x1={1}".format("Metrics_by_support", "all"))
        for i in range(2):
            support_plots(metrics_by_xgroup[i], labels, "{0}_x1={1}".format("Metrics_by_support", str(i)))