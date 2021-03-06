from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext_modules = [Extension("malis.malis",
                         ["malis/malis.pyx", "malis/malis_cpp.cpp"],
                         language='c++',
                         extra_compile_args=["-std=c++14"]),]

setup(name="Malis",
      cmdclass={'build_ext': build_ext},
      packages=["malis"],
      include_dirs=[numpy.get_include(), '/usr/local/include'],
      ext_modules=ext_modules)
