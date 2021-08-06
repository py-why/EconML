from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import os
import re
from glob import glob
from pathlib import Path

with open(os.path.join(os.path.dirname(__file__), "econml", "_version.py")) as file:
    for line in file:
        m = re.fullmatch("__version__ = '([^']+)'\n", line)
        if m:
            version = m.group(1)

pyx_files = glob("econml/**/*.pyx", recursive=True)
c_files = glob("econml/**/*.c", recursive=True)

# If both a .pyx and a .c file exist, we assume the .c file is up to date and don't force a recompile
pyx_files = [file for file in pyx_files if (os.path.splitext(file)[0] + ".c") not in c_files]

c_extensions = [Extension(os.path.splitext(file)[0].replace(os.sep, '.'),
                          [file],
                          include_dirs=[np.get_include()])
                for file in c_files]

if pyx_files:
    from Cython.Build import cythonize
    pyx_extensions = cythonize([Extension("*",
                                          pyx_files,
                                          include_dirs=[np.get_include()])],
                               language_level="3")
else:
    pyx_extensions = []
# configuration is all pulled from setup.cfg
setup(ext_modules=c_extensions + pyx_extensions,
      zip_safe=False,
      version=version)
