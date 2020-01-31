# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import argparse
from mcpy.monte_carlo import MonteCarlo, MonteCarloSweep
import importlib

def monte_carlo_main():
    """
    Entry point to run monte carlo simulations with specifications from a config file
    run this file with command line arguments '--config [config_file]'
    """
    parser = argparse.ArgumentParser(description='Process a config file for the monte carlo simulation.')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args(sys.argv[1:])
    config = importlib.import_module(args.config, __name__)

    print("Running monte carlo simulation from {}.".format(args.config))
    # one may want to define new config types
    # the sweep over the parameters could be done differently depending on type
    # imagine, a different MonteCarloSweep function
    # or where MonteCarloSweep sweeps over the parameters differently
    # depending on the type specified
    if config.CONFIG['type'] == 'single_parameter':
        MonteCarlo(config.CONFIG).run()
    elif config.CONFIG['type'] == 'sweep_parameter':
        MonteCarloSweep(config.CONFIG).run()

if __name__=="__main__":
    monte_carlo_main()
