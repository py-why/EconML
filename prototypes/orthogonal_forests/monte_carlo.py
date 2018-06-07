import argparse
import matplotlib
matplotlib.use('Agg')
import os
import subprocess
import sys
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from hetero_dml import HeteroDML, ForestHeteroDML
from ortho_forest import OrthoForest, DishonestOrthoForest
from residualizer import dml, second_order_dml

def piecewise_linear_te(x):
    if x[0] <= .3:
        return x[0]  + 2
    elif x[0] <= .6:
        return 6 * (x[0]-0.3) + piecewise_linear_te([.3]) 
    else:
       return  - 3 * (x[0]-0.6) + piecewise_linear_te([.6])

def step_te(x):
    if x[0] < .2:
        return 1
    elif x[0] < .6:
        return 5 
    else:
       return  3

def polynomial_te(x):
    if x[0] < .2:
        return 3 * (x[0]**2)
    elif x[0] < .6:
        return 3 * (x[0]**2) + 1
    else:
       return  6 * x[0] + 2

def doublez_te(x):
    if x[1] == 0:
        return piecewise_linear_te(x)
    else:
        return step_te(x)

if __name__ == "__main__":
    ####################
    # Argument parsing #
    ####################
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_experiments", type=int, help="number of experiments", default=10)
    parser.add_argument("--n_samples", type=int, help="number of samples", default=5000)
    parser.add_argument("--n_dim", type=int, help="number of controls", default=100)
    parser.add_argument("--support_size", type=int, help="size of control support", default=5)
    parser.add_argument("--n_x", type=int, help="number of covariates", default=1)
    parser.add_argument("--seed", type=int, help="number generator seed", default=12345)
    parser.add_argument("--te_func", type=int, help="treatment effect function indicator", default=0)
    parser.add_argument("--n_trees", type=int, help="number of trees in ortho forest", default=100)
    parser.add_argument("--max_splits", type=int, help="maximum splits in ortho forest", default=20)
    parser.add_argument("--min_leaf_size", type=int, help="min leaf size in ortho forest", default=5)
    parser.add_argument("--bootstrap", type=int, help="use bootstrap subsampling or subsampling", default=0)
    parser.add_argument("--subsample_power", type=float, help="what power of the original data set is the subsample size", default=.88)
    parser.add_argument("--output_dir", type=str, help="directory for saving results", default=".")
    parser.add_argument("--method_id", type=int, help="method_id", default=0)
    parser.add_argument("--control_for_W", type=int, help="control for W", default=1)
    parser.add_argument("--R", action='store_true', help="Whether to run R simulations")
    parser.add_argument("--P", action='store_true', help="Whether to plot results")
    
    method_names = ['OrthoForest', 'OrthoForestCV', 'DishonestOrthoForestCV', 'HeteroDML', 'ForestHeteroDML']
    args = parser.parse_args(sys.argv[1:])
    np.random.seed(args.seed)
    file_prefix_rel = "{class_name}_n_samples_{n_samples}_n_dim_{n_dim}_n_x_{n_x}_support_{support_size}_seed_{seed}_n_trees_{n_trees}_max_splits_{max_splits}_min_leaf_size_{min_leaf_size}_{sampling}_".format(
                        class_name=("NotControlling" if args.control_for_W==0 else "") + method_names[args.method_id],
                        n_samples=args.n_samples,
                        n_dim=args.n_dim,
                        n_x=args.n_x,
                        support_size=args.support_size,
                        seed=args.seed,
                        n_trees=args.n_trees,
                        max_splits=args.max_splits,
                        min_leaf_size=args.min_leaf_size,
                        sampling=("bootstrap" if bool(args.bootstrap) else "s_pow_{s_pow}".format(s_pow="{:.2}".format(args.subsample_power).replace('.','_')))
                        )    
    file_prefix = os.path.join(args.output_dir, file_prefix_rel)
    te_func = None
    if args.te_func == 0:
        te_func = piecewise_linear_te
    elif args.te_func == 1:
        te_func = step_te
    elif args.te_func == 2:
        te_func = polynomial_te
    elif args.te_func == 3:
        assert(args.n_x > 1)
        te_func = doublez_te
    else:
        print("UNKNOWN TREATMENT EFFECT FUNCTION")
        exit()

    ##########################################
    # Parameters constant across experiments #
    ##########################################
    # Outcome support
    support_Y = np.random.choice(range(args.n_dim), size=args.support_size, replace=False)
    coefs_Y = np.random.uniform(0, 1, size=args.support_size)
    sigma_Y = 1
    epsilon_sample = lambda n: np.random.uniform(-sigma_Y, sigma_Y, size=n)
    # Treatment support 
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=args.support_size)
    sigma_T = 1
    eta_sample = lambda n: np.random.uniform(-sigma_T, sigma_T, size=n) 
    # Evaluation grid
    if args.n_x > 1:
        x_grid = np.concatenate(
            (
                np.column_stack((np.arange(0, 1, 0.01), np.zeros(100))), 
                np.column_stack((np.arange(0, 1, 0.01), np.ones(100)))
            )
        )
    else:
        x_grid = np.array(list(product(np.arange(0, 1, 0.01), repeat=args.n_x)))
    # Treatment effects array
    treatment_effects = np.zeros((x_grid.shape[0], args.n_experiments))
    # Other variables
    file_header = ",".join(["W"+str(i) for i in range(args.n_dim)] + ["x"+str(i) for i in range(args.n_x)] + ["T", "Y", 
    "res_T_W", "res_T_Wx", "res_Y_W", "res_Y_Wx"])
    names_fn = "{file_prefix}file_names.txt".format(file_prefix=file_prefix)
    names_f = open(names_fn,"w+")
    names_f.write("{0}\n".format(args.n_experiments))

    ###########################
    # Data Generating Process #
    ###########################
    start = time.time()
    for t in range(args.n_experiments):
        # Log iteration
        print("Iteration {}".format(t))
        # Generate controls, features, treatment and outcome
        W = np.random.normal(0, 1, size=(args.n_samples, args.n_dim))
        if args.n_x > 1:
            x = np.concatenate(
                (np.random.uniform(0, 1, size=(args.n_samples, 1)), 
                np.random.binomial(1, 0.5, size=(args.n_samples, args.n_x-1))),
                axis=1
                )
        else:
            x = np.random.uniform(0, 1, size=(args.n_samples, args.n_x))
        TE = np.array([te_func(x_i) for x_i in x]) # Heterogeneous treatment effects
        T = np.dot(W[:, support_T], coefs_T) + eta_sample(args.n_samples)
        Y = TE * T + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(args.n_samples)
        # T and Y residuals to be used in later scripts
        res_model1 = LassoCV()
        res_model2 = LassoCV()
        mid = args.n_samples // 2
        Wx = np.concatenate((W,x), axis=1)
        res_T_W = np.concatenate((T[:mid] - res_model1.fit(W[mid:], T[mid:]).predict(W[:mid]),
                                   T[mid:] - res_model2.fit(W[:mid], T[:mid]).predict(W[mid:])))
        res_T_Wx = np.concatenate((T[:mid] - res_model1.fit(Wx[mid:], T[mid:]).predict(Wx[:mid]),
                                   T[mid:] - res_model2.fit(Wx[:mid], T[:mid]).predict(Wx[mid:])))
        res_Y_W = np.concatenate((Y[:mid] - res_model1.fit(W[mid:], Y[mid:]).predict(W[:mid]),
                                   Y[mid:] - res_model2.fit(W[:mid], Y[:mid]).predict(W[mid:])))
        res_Y_Wx = np.concatenate((Y[:mid] - res_model1.fit(Wx[mid:], Y[mid:]).predict(Wx[:mid]),
                                   Y[mid:] - res_model2.fit(Wx[:mid], Y[:mid]).predict(Wx[mid:])))
        
        # Save generated dataset
        file_name = "{file_prefix}experiment_{t}.csv".format(file_prefix=file_prefix, t=t)
        names_f.write(file_name + "\n")
        np.savetxt(file_name, np.concatenate((
            W, x, T.reshape(-1, 1), Y.reshape(-1, 1),
            res_T_W.reshape(-1, 1), res_T_Wx.reshape(-1, 1),
            res_Y_W.reshape(-1, 1), res_Y_Wx.reshape(-1, 1)
        ), axis=1), 
        delimiter=",", header=file_header, comments='')

        ##################
        # ORF parameters #
        ##################
        n_trees = args.n_trees
        max_splits = args.max_splits
        min_leaf_size = args.min_leaf_size
        bootstrap = (args.bootstrap == 1)
        subsample_ratio = ((args.n_samples/np.log(args.n_dim))**(args.subsample_power)) / (args.n_samples)
        print("Subsample_ratio: {}".format(subsample_ratio))
        residualizer = dml
        if bootstrap:
            lambda_reg = np.sqrt(np.log(args.n_dim) / (10 * args.n_samples))
        else:
            lambda_reg = np.sqrt(np.log(args.n_dim) / (10 * subsample_ratio * args.n_samples))
        print("Lambda: {}".format(lambda_reg))
        model_T = Lasso(alpha=lambda_reg)
        model_Y = Lasso(alpha=lambda_reg)

        #######################################
        # Train and evaluate treatment effect #
        #######################################
        if args.method_id == 0:
            est = OrthoForest(n_trees=n_trees, min_leaf_size=min_leaf_size, residualizer=residualizer,
                    max_splits=max_splits, subsample_ratio=subsample_ratio, bootstrap=bootstrap, 
                    model_T=model_T, model_Y=model_Y)
        elif args.method_id == 1:
            est = OrthoForest(n_trees=n_trees, min_leaf_size=min_leaf_size, residualizer=residualizer,
                      max_splits=max_splits, subsample_ratio=subsample_ratio, bootstrap=bootstrap, 
                      model_T=model_T, model_Y=model_Y, model_T_final=LassoCV(), model_Y_final=LassoCV())
        elif args.method_id == 2:
            est = DishoestOrthoForest(n_trees=n_trees, min_leaf_size=min_leaf_size, residualizer=residualizer,
                      max_splits=max_splits, subsample_ratio=subsample_ratio, bootstrap=bootstrap, 
                      model_T=model_T, model_Y=model_Y, model_T_final=LassoCV(), model_Y_final=LassoCV())
        elif args.method_id == 3:
            est = HeteroDML(poly_degree=3)
        elif args.method_id == 4:
            est = HeteroDML(poly_degree=3, model_T=RandomForestRegressor(), model_Y=RandomForestRegressor())
        else:
            print("UNKNOWN METHOD")
            exit()
        if args.control_for_W:
            print("Controlling for W")
            est.fit(W, x, T, Y)
        else:
            est.fit(np.zeros((W.shape[0], 1)), x, T, Y)
        treatment_effects[:, t] = est.predict(x_grid)
        
    #########
    # Plots #
    #########
    if args.te_func == 3:
        fig = plt.figure(figsize=(10, 4))
        titles = ["$x_1 = 0$", "$x_1 = 1$"]
        for i, x_slice in enumerate([(x_grid[:, 1] == 0), (x_grid[:, 1] == 1)]):
            ax = fig.add_subplot(1, 2, i+1)
            tes = treatment_effects[x_slice]
            y = np.mean(tes, axis=1)
            err_up = np.percentile(tes, 95, axis=1)
            err_bottom = np.percentile(tes, 5, axis=1)
            ax.plot(x_grid[x_slice][:, 0], y, label='Mean estimate')
            ax.fill_between(x_grid[x_slice][:, 0], err_up, err_bottom, alpha=0.5)
            TE_grid = np.array([te_func(x_i) for x_i in x_grid[x_slice]])
            ax.plot(x_grid[x_slice][:, 0], TE_grid, 'b--', label='True effect')
            ax.set_ylabel("Treatment effect")
            ax.set_xlabel("$x_0$")
            ax.set_title(titles[i])
            ax.legend()
        fig.savefig("{file_prefix}_true_vs_pred_high_res.png".format(file_prefix=file_prefix), dpi=300, bbox_inches='tight')
        fig.savefig("{file_prefix}_true_vs_pred.png".format(file_prefix=file_prefix), bbox_inches='tight')
        fig.savefig("{file_prefix}_true_vs_pred.pdf".format(file_prefix=file_prefix), bbox_inches='tight')
    else:
        y = np.mean(treatment_effects, axis=1)
        err_up = np.percentile(treatment_effects, 95, axis=1)
        err_bottom = np.percentile(treatment_effects, 5, axis=1)
        plt.plot(x_grid[:, 0], y, label='Mean estimate')
        plt.fill_between(x_grid[:, 0], err_up, err_bottom, alpha=0.5)
        TE_grid = np.array([te_func(x_i) for x_i in x_grid])
        plt.plot(x_grid[:, 0], TE_grid, 'b--', label='True effect')
        plt.ylabel("Treatment effect")
        plt.xlabel("x")
        plt.legend()
        plt.savefig("{file_prefix}_true_vs_pred_high_res.png".format(file_prefix=file_prefix), dpi=300, bbox_inches='tight')
        plt.savefig("{file_prefix}_true_vs_pred.png".format(file_prefix=file_prefix), bbox_inches='tight')
        plt.savefig("{file_prefix}_true_vs_pred.pdf".format(file_prefix=file_prefix), bbox_inches='tight')

    ################
    # Save results #
    ################
    results_file_name = "{file_prefix}results.csv".format(file_prefix=file_prefix)
    TE_grid = np.array([te_func(x_i) for x_i in x_grid])
    aux = np.ones(args.n_experiments)
    W_dummy = np.random.normal(0, 1, size=(x_grid.shape[0], args.n_dim)) # For comparison with models that take in W U x
    np.savetxt(results_file_name, np.concatenate((x_grid, W_dummy, TE_grid.reshape(-1, 1), treatment_effects), axis=1), 
    delimiter=",", header=",".join(['x'+str(i) for i in range(args.n_x)] + ['W'+str(i) for i in range(args.n_dim)] + ['TE_hat'] + ["TE_"+str(i) for i in range(args.n_experiments)]), comments='')

    end = time.time()
    names_f.write("Runtime: {0} minutes\n".format(int((end-start)/60)))
    names_f.close()

    ###############
    # Run Rscript #
    ###############
    if args.R:
        subprocess.check_call("Rscript GRF_treatment_effects.R --prefix {0}".format(file_prefix), shell=True)