import sys
import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('econml')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
