"""
Utilities useful during the build.
"""

import os
import sklearn
import contextlib

from distutils.version import LooseVersion

CYTHON_MIN_VERSION = '0.28.5'


def _check_cython_version():
    message = ('Please install Cython with a version >= {0} in order '
               'to build a scikit-learn from source.').format(
        CYTHON_MIN_VERSION)
    try:
        import Cython
    except ModuleNotFoundError:
        # Re-raise with more informative error message instead:
        raise ModuleNotFoundError(message)

    if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
        message += (' The current version of Cython is {} installed in {}.'
                    .format(Cython.__version__, Cython.__path__))
        raise ValueError(message)


def cythonize_extensions(top_path, config):
    """Check that a recent Cython is available and cythonize extensions"""
    _check_cython_version()
    from Cython.Build import cythonize

    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': 3})
