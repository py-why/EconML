from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
import re

with open(os.path.join(os.path.dirname(__file__), "econml", "__init__.py")) as file:
    for line in file:
        m = re.fullmatch("__version__ = '([^']+)'\n", line)
        if m:
            version = m.group(1)

# configuration is all pulled from setup.cfg
setup(ext_modules=cythonize([Extension("*", ["**/*.pyx"],
                                       include_dirs=[np.get_include()])],
                            language_level="3"),
      zip_safe=False,
      version=version)
