from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'ext_learn',
  ext_modules = cythonize("external_learner.py"),
)