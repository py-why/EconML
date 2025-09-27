#!/usr/bin/env env python3
"""
Treatment Effect Estimation Results Analysis and Visualization

This module analyzes and visualizes results from various treatment effect estimation methods,
including bias, variance, RMSE, and R² comparisons across different experimental conditions.
"""

import argparse
import copy
import itertools
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import rcParams
from sklearn.metrics import r2_score

# Configure matplotlib
matplotlib.rcParams['font.family'] = "serif"

###################
# Constants       #
###################

PLOT_CONTROLS = ["support"]
LABEL_ORDER = ["ORF-CV", "ORF", "GRF-xW", "GRF-x", "GRF-Res", "HeteroDML-Lasso", "HeteroDML-RF"]
METHOD_MAPPING = {
    "OrthoForestCV": "ORF-CV",
    "OrthoForest": "ORF", 
    "GRF_Wx": "GRF-xW",
    "GRF_x": "GRF-x",
    "GRF_res_Wx": "GRF-Res",
    "HeteroDML": "HeteroDML-Lasso",
    "ForestHeteroDML": "HeteroDML-RF"
}
CORRESPONDING_STR = list(METHOD_MAPPING.keys())

# Plot configuration
FIGURE_SIZE_JOINT = (10, 5)
FIGURE_SIZE_METRICS = (12, 3)
DPI_HIGH_RES = 300
PERCENTILE_UPPER = 95
PERCENTILE_LOWER = 5

# Color schemes
COLOR_INDICES = [0, 3, 12, 14, 15, 4, 6]

# Output directories
OUTPUT_SUBDIRS = ["jpg_low_res", "jpg_high_res", "pdf_low_res"]

###################
# Data Classes    #
###################

class MetricResults:
    """Container for metric calculation results."""
    
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

class ExperimentResults:
    """Container for all experimental results."""
    
    def __init__(self):
        self.bias = None
        self.variance = None
        self.rmse = None
        self.r2 = None

###################
# File Operations #
###################

class FileProcessor:
    """Handles file operations and data loading."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._ensure_output_dirs()
    
    def _ensure_output_dirs(self) -> None:
        """Create necessary output directories."""
        for subdir in OUTPUT_SUBDIRS:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def has_plot_controls(self, fname: str, control_combination: List[str]) -> bool:
        """Check if filename contains all required control parameters."""
        return all(f"_{control}_" in fname for control in control_combination)
    
    def get_file_key(self, fname: str) -> str:
        """Extract file key for grouping related files."""
        if "GRF" in fname:
            return "_" + "_".join(re.split("GRF_", fname)[0].split("_")[1:])
        else:
            return "_" + "_".join(re.split("results", fname)[0].split("_")[1:])
    
    def sort_filenames(self, file_names: List[str]) -> Tuple[List[str], np.ndarray]:
        """Sort filenames according to predefined method order."""
        sorted_file_names = []
        label_indices = []
        
        for i, method_str in enumerate(CORRESPONDING_STR):
            for fname in file_names:
                if self._matches_method(fname, method_str):
                    sorted_file_names.append(fname)
                    label_indices.append(i)
                    break
        
        return sorted_file_names, np.array(LABEL_ORDER)[label_indices]
    
    def _matches_method(self, fname: str, method_str: str) -> bool:
        """Check if filename matches a specific method."""
        if "GRF" not in fname:
            return fname.split("_")[0] == method_str
        else:
            return f"_{method_str}_" in fname
    
    def get_file_groups(self, agg_fnames: List[str]) -> Tuple[Dict[str, List[str]], np.ndarray]:
        """Group files by experimental conditions."""
        all_file_names = {}
        control_values = self._extract_control_values(agg_fnames)
        control_combinations = list(itertools.product(*control_values))
        
        final_labels = None
        for control_combination in control_combinations:
            file_names = [f for f in agg_fnames 
                         if self.has_plot_controls(f, control_combination)]
            
            if file_names:
                file_key = self.get_file_key(file_names[0])
                sorted_names, labels = self.sort_filenames(file_names)
                all_file_names[file_key] = sorted_names
                if final_labels is None:
                    final_labels = labels
        
        return all_file_names, final_labels
    
    def _extract_control_values(self, agg_fnames: List[str]) -> List[List[str]]:
        """Extract unique control parameter values from filenames."""
        control_values = []
        for control in PLOT_CONTROLS:
            vals = set()
            control_prefix = f"{control}_"
            
            for fname in agg_fnames:
                match = re.search(f"{control_prefix}(\\d+)", fname)
                if match:
                    vals.add(f"{control_prefix}{match.group(1)}")
            
            control_values.append(list(vals))
        
        return control_values
    
    def merge_results(self, sf: Tuple[str, str], split_files_seeds: Dict) -> None:
        """Merge results from multiple seed runs."""
        name_template = "{0}seed_{1}_{2}"
        seeds = split_files_seeds[sf]
        
        try:
            # Load first file
            first_file = self.input_dir / name_template.format(sf[0], seeds[0], sf[1])
            df = pd.read_csv(first_file)
            
            te_idx = len([c for c in df.columns if re.search("TE_[0-9]", c)])
            
            # Merge additional seeds
            for seed in seeds[1:]:
                seed_file = self.input_dir / name_template.format(sf[0], seed, sf[1])
                new_df = pd.read_csv(seed_file)
                te_cols = [c for c in new_df.columns if re.search("TE_[0-9]", c)]
                
                for te_col in te_cols:
                    df[f"TE_{te_idx}"] = new_df[te_col]
                    te_idx += 1
            
            # Save merged results
            agg_fname = self.output_dir / f"{sf[0]}{sf[1]}"
            df.to_csv(agg_fname, index=False)
            
        except Exception as e:
            print(f"Error merging results for {sf}: {e}")
            raise
    
    def get_results(self, fname: str) -> pd.DataFrame:
        """Load and filter results data."""
        try:
            df = pd.read_csv(self.output_dir / fname)
            x_cols = [c for c in df.columns if "x" in c]
            te_cols = [c for c in df.columns if "TE_" in c]
            return df[x_cols + te_cols]
        except Exception as e:
            print(f"Error loading results from {fname}: {e}")
            raise

###################
# Analysis        #
###################

class MetricsCalculator:
    """Calculates performance metrics for treatment effect estimation."""
    
    @staticmethod
    def calculate_r2(df: pd.DataFrame) -> np.ndarray:
        """Calculate R² scores for all treatment effect columns."""
        te_cols = [c for c in df.columns if re.search('TE_[0-9]+', c)]
        return np.array([r2_score(df["TE_hat"], df[col]) for col in te_cols])
    
    @staticmethod
    def calculate_metrics(dfs: List[pd.DataFrame]) -> ExperimentResults:
        """Calculate bias, variance, RMSE, and R² for all dataframes."""
        n_obs = len(dfs[0])
        n_methods = len(dfs)
        
        biases = np.zeros((n_obs, n_methods))
        variances = np.zeros((n_obs, n_methods))
        rmses = np.zeros((n_obs, n_methods))
        r2_scores = []
        
        for i, df in enumerate(dfs):
            te_cols = [c for c in df.columns if re.search('TE_[0-9]+', c)]
            treatment_effects = df[te_cols]
            
            # Calculate metrics
            mean_te = np.mean(treatment_effects, axis=1)
            biases[:, i] = np.abs(mean_te - df["TE_hat"])
            variances[:, i] = np.std(treatment_effects, axis=1)
            rmses[:, i] = np.mean(((treatment_effects.T - df["TE_hat"].values).T)**2, axis=1)
            r2_scores.append(MetricsCalculator.calculate_r2(df))
        
        # Create results object
        results = ExperimentResults()
        results.bias = MetricResults(np.mean(biases, axis=0), np.std(biases, axis=0))
        results.variance = MetricResults(np.mean(variances, axis=0), np.std(variances, axis=0))
        results.rmse = MetricResults(np.mean(rmses, axis=0), np.std(rmses, axis=0))
        results.r2 = MetricResults(
            np.array([np.mean(r2_scores[i]) for i in range(len(r2_scores))]),
            np.array([np.std(r2_scores[i]) for i in range(len(r2_scores))])
        )
        
        return results

###################
# Visualization   #
###################

class PlotGenerator:
    """Generates various types of plots for results visualization."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
    
    def save_plots(self, fig: plt.Figure, fname: str, lgd: Optional[Any] = None) -> None:
        """Save figure in multiple formats."""
        save_kwargs = {'bbox_inches': 'tight'}
        if lgd is not None:
            save_kwargs['bbox_extra_artists'] = (lgd,)
        
        # Save in different formats and resolutions
        fig.savefig(self.output_dir / "jpg_low_res" / f"{fname}.png", **save_kwargs)
        fig.savefig(self.output_dir / "jpg_high_res" / f"{fname}.png", 
                   dpi=DPI_HIGH_RES, **save_kwargs)
        fig.savefig(self.output_dir / "pdf_low_res" / f"{fname}.pdf", **save_kwargs)
    
    def create_joint_plots(self, file_key: str, dfs: List[pd.DataFrame], 
                          labels: List[str], file_name_prefix: str) -> None:
        """Create joint treatment effect plots."""
        n_methods = len(dfs)
        n_cols = min(4, n_methods)
        n_rows = int(np.ceil(n_methods / n_cols))
        
        fig = plt.figure(figsize=FIGURE_SIZE_JOINT)
        ymax = max([df["TE_hat"].max() for df in dfs]) + 1
        
        for i, df in enumerate(dfs):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            te_cols = [c for c in df.columns if re.search('TE_[0-9]+', c)]
            treatment_effects = df[te_cols]
            
            y_mean = np.mean(treatment_effects, axis=1)
            err_up = np.percentile(treatment_effects, PERCENTILE_UPPER, axis=1)
            err_bottom = np.percentile(treatment_effects, PERCENTILE_LOWER, axis=1)
            
            ax.fill_between(df["x0"], err_up, err_bottom, alpha=0.5)
            
            if i == 0:
                ax.plot(df["x0"], y_mean, label='Mean estimate')
                ax.plot(df["x0"], df["TE_hat"], 'b--', label='True effect')
            else:
                ax.plot(df["x0"], y_mean)
                ax.plot(df["x0"], df["TE_hat"], 'b--')
            
            if i % n_cols == 0:
                ax.set_ylabel("Treatment effect")
            
            ax.set_ylim(ymax=ymax)
            ax.set_title(labels[i])
            
            if i + 1 > n_cols * (n_rows - 1):
                ax.set_xlabel("x")
        
        fig.legend(loc=(0.8, 0.25))
        fig.tight_layout()
        self.save_plots(fig, file_name_prefix)
        plt.close(fig)
    
    def create_metrics_plots(self, file_key: str, dfs: List[pd.DataFrame], 
                           labels: List[str], file_name_prefix: str) -> None:
        """Create violin plots for bias, variance, and RMSE metrics."""
        metrics = ["bias", "variance", "rmse"]
        fig = plt.figure(figsize=FIGURE_SIZE_METRICS)
        
        violin_bodies = []
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(1, len(metrics), i + 1)
            bodies = self._create_metric_subplot(dfs, ax, metric)
            if i == 0:
                violin_bodies = bodies
        
        lgd = fig.legend(violin_bodies, labels, ncol=len(labels), 
                        loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        self.save_plots(fig, file_name_prefix, lgd)
        plt.close(fig)
    
    def _create_metric_subplot(self, dfs: List[pd.DataFrame], ax: plt.Axes, 
                             metric: str) -> List[Any]:
        """Create subplot for a specific metric."""
        palette = plt.get_cmap('Set1')
        
        if metric == "bias":
            data = self._calculate_bias_data(dfs)
            ax.set_title("Bias")
        elif metric == "variance":
            data = self._calculate_variance_data(dfs)
            ax.set_title("Variance")
        elif metric == "rmse":
            data = self._calculate_rmse_data(dfs)
            ax.set_title("RMSE")
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        vparts = ax.violinplot(data, showmedians=True)
        ax.set_xticks([])
        
        # Style violin plots
        for i, body in enumerate(vparts['bodies']):
            color_idx = i if i < 5 else i + 1
            body.set_facecolor(palette(color_idx))
            body.set_edgecolor(palette(color_idx))
            body.set_alpha(0.9)
        
        # Style other violin plot elements
        for element in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            if element in vparts:
                vparts[element].set_color('black')
                vparts[element].set_alpha(0.7 if element != 'cbars' else 0.3)
                if element == 'cbars':
                    vparts[element].set_linestyle('--')
        
        return vparts['bodies']
    
    def _calculate_bias_data(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        """Calculate bias data for violin plot."""
        biases = np.zeros((len(dfs[0]), len(dfs)))
        for i, df in enumerate(dfs):
            te_cols = [c for c in df.columns if re.search('TE_[0-9]+', c)]
            treatment_effects = df[te_cols]
            biases[:, i] = np.abs(np.mean(treatment_effects, axis=1) - df["TE_hat"])
        return biases
    
    def _calculate_variance_data(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        """Calculate variance data for violin plot."""
        variances = np.zeros((len(dfs[0]), len(dfs)))
        for i, df in enumerate(dfs):
            te_cols = [c for c in df.columns if re.search('TE_[0-9]+', c)]
            treatment_effects = df[te_cols]
            variances[:, i] = np.std(treatment_effects, axis=1)
        return variances
    
    def _calculate_rmse_data(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        """Calculate RMSE data for violin plot."""
        rmses = np.zeros((len(dfs[0]), len(dfs)))
        for i, df in enumerate(dfs):
            te_cols = [c for c in df.columns if re.search('TE_[0-9]+', c)]
            treatment_effects = df[te_cols]
            rmses[:, i] = np.mean(((treatment_effects.T - df["TE_hat"].values).T)**2, axis=1)
        return rmses
    
    def create_support_plots(self, all_metrics: Dict, labels: List[str], 
                           file_name_prefix: str) -> None:
        """Create plots showing metrics vs support size."""
        palette = plt.get_cmap('Set1')
        x_values = sorted(all_metrics.keys())
        metrics = ["bias", "variance", "rmse"]
        titles = ["Bias", "Variance", "RMSE"]
        
        fig = plt.figure(figsize=FIGURE_SIZE_METRICS)
        plot_objects = []
        
        for metric_idx, metric in enumerate(metrics):
            ax = fig.add_subplot(1, len(metrics), metric_idx + 1)
            
            for i, label in enumerate(labels):
                color_idx = i if i < 5 else i + 1
                
                # Extract metric values across support sizes
                err_values = np.array([all_metrics[x][metric].std[i] for x in x_values])
                mean_values = np.array([all_metrics[x][metric].mean[i] for x in x_values])
                
                # Plot with error bands
                fill = ax.fill_between(x_values, mean_values - err_values/6, 
                                     mean_values + err_values/6, 
                                     alpha=0.5, color=palette(color_idx))
                ax.plot(x_values, mean_values, label=label, color=palette(color_idx))
                
                if metric_idx == 0:
                    plot_obj = copy.copy(fill)
                    plot_obj.set_alpha(1.0)
                    plot_objects.append(plot_obj)
            
            ax.set_title(titles[metric_idx])
            ax.set_xlabel("Support size")
        
        lgd = fig.legend(plot_objects, labels, ncol=len(labels), 
                        loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        self.save_plots(fig, file_name_prefix, lgd)
        plt.close(fig)

###################
# Main Analysis   #
###################

def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyze treatment effect estimation results")
    parser.add_argument("--output_dir", type=str, default=".", 
                       help="Directory for saving results")
    parser.add_argument("--input_dir", type=str, default=".", 
                       help="Directory containing input files")
    parser.add_argument("--merge", action='store_true', 
                       help="Merge results from multiple seeds")
    
    args = parser.parse_args()
    
    # Initialize processors
    file_processor = FileProcessor(args.input_dir, args.output_dir)
    plot_generator = PlotGenerator(args.output_dir)
    
    # Process files
    all_files = os.listdir(args.input_dir)
    results_files = [f for f in all_files if f.endswith("results.csv") and "seed" in f]
    
    split_files = set([
        (re.split("seed_[0-9]+_", f)[0], re.split("seed_[0-9]+_", f)[1]) 
        for f in results_files
    ])
    
    split_files_seeds = {
        k: [int(re.search("seed_(\\d+)_", f).group(1)) 
            for f in results_files if f.startswith(k[0]) and f.endswith(k[1])]
        for k in split_files
    }
    
    agg_fnames = [f"{sf[0]}{sf[1]}" for sf in split_files]
    
    # Merge results if requested
    if args.merge:
        print("Merging results from multiple seeds...")
        Parallel(n_jobs=-1, verbose=3)(
            delayed(file_processor.merge_results)(sf, split_files_seeds) 
            for sf in split_files
        )
    
    # Group files and generate plots
    agg_file_groups, labels = file_processor.get_file_groups(agg_fnames)
    
    all_metrics = {}
    metrics_by_xgroup = [{}, {}]
    
    for group_key in agg_file_groups:
        print(f"Processing group: {group_key}")
        agg_file_group = agg_file_groups[group_key]
        dfs = [file_processor.get_results(fname) for fname in agg_file_group]
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_metrics(dfs)
        support_size = int(re.search("support_(\\d+)", group_key).group(1))
        all_metrics[support_size] = {
            "bias": metrics.bias,
            "variance": metrics.variance, 
            "rmse": metrics.rmse,
            "r2": metrics.r2
        }
        
        # Determine feature dimensionality
        n_features = len([c for c in dfs[0].columns if re.search("x[0-9]", c)])
        
        if n_features == 1:
            # Single feature case
            plot_generator.create_joint_plots(
                group_key, dfs, labels, f"Example{group_key}"
            )
            plot_generator.create_metrics_plots(
                group_key, dfs, labels, f"Metrics{group_key}"
            )
        else:
            # Multiple feature case
            plot_generator.create_metrics_plots(
                group_key, dfs, labels, f"Metrics{group_key}_x1=all"
            )
            
            # Create plots for each feature group
            for i in range(2):
                dfs_subset = [df[df["x1"] == i] for df in dfs]
                plot_generator.create_joint_plots(
                    group_key, dfs_subset, labels, f"Example{group_key}_x1={i}"
                )
                plot_generator.create_metrics_plots(
                    group_key, dfs_subset, labels, f"Metrics{group_key}_x1={i}"
                )
                
                # Store metrics for support plots
                subset_metrics = MetricsCalculator.calculate_metrics(dfs_subset)
                metrics_by_xgroup[i][support_size] = {
                    "bias": subset_metrics.bias,
                    "variance": subset_metrics.variance,
                    "rmse": subset_metrics.rmse,
                    "r2": subset_metrics.r2
                }
    
    # Generate support size comparison plots
    if n_features == 1:
        plot_generator.create_support_plots(all_metrics, labels, "Metrics_by_support")
    else:
        plot_generator.create_support_plots(all_metrics, labels, "Metrics_by_support_x1=all")
        for i in range(2):
            plot_generator.create_support_plots(
                metrics_by_xgroup[i], labels, f"Metrics_by_support_x1={i}"
            )
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
