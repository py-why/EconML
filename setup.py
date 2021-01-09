from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np


# configuration is all pulled from setup.cfg
setup(ext_modules=cythonize([Extension("*", ["**/*.pyx"],
                                       include_dirs=[np.get_include()])],
                            language_level="3"),
      zip_safe=False)
