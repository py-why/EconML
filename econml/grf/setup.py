import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("grf", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension("_criterion",
                         sources=["_criterion.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_utils",
                         sources=["_utils.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
