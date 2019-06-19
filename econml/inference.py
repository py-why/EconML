# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Options for performing inference in estimators."""


class BootstrapOptions:
    """
    Wrapper storing bootstrap options.

    This class can be used for inference with any CATE estimator.

    Parameters
    ----------
    n_bootstrap_samples : int, optional (default 100)
        How many draws to perform.

    n_jobs: int, optional (default -1)
        The maximum number of concurrently running jobs, as in joblib.Parallel.

    """

    def __init__(self, n_bootstrap_samples=100, n_jobs=-1):
        self._n_bootstrap_samples = n_bootstrap_samples
        self._n_jobs = n_jobs

    @property
    def n_bootstrap_samples(self):
        """Get how many draws to perform."""
        return self._n_bootstrap_samples

    @property
    def n_jobs(self):
        """Get the maximum number of concurrently running jobs, as in joblib.Parallel."""
        return self._n_jobs
