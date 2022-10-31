import os
import shlex
import platform
import re
import subprocess
import sys
from itertools import chain
from distutils.version import LooseVersion

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

if sys.version_info <= (3, 8):
    print("Python 3.8 or higher required, please upgrade.")
    sys.exit(1)

VERSION = "0.6.0"

REQUIREMENTS = ["numpy>=1.21", "fenics-dolfinx>=0.6.0.dev0"]

extras = {
    'docs': ['jupyter-book'],
}

# 'all' includes all of the above
extras['all'] = list(chain(*extras.values()))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the"
                               + "following extensions:"
                               + ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
        cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                       '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        env = os.environ.copy()
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        # default to 3 build threads
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in env:
            env["CMAKE_BUILD_PARALLEL_LEVEL"] = "3"

        import pybind11
        env['pybind11_DIR'] = pybind11.get_cmake_dir()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp, env=env)


setup(name='dolfinx-mpc',
      version=VERSION,
      author='Jørgen S. Dokken',
      description='Python interface for multipointconstraints in dolfinx',
      long_description='',
      packages=["dolfinx_mpc", "dolfinx_mpc.utils", "dolfinx_mpc.numba"],
      ext_modules=[CMakeExtension('dolfinx_mpc.cpp')],
      package_data={'dolfinx_mpc.wrappers': ['*.h'], 'dolfinx_mpc': ['py.typed'],
                    'dolfinx_mpc.numba': ['py.typed'], 'dolfinx_mpc.utils': ['py.typed']},
      cmdclass=dict(build_ext=CMakeBuild),
      install_requires=REQUIREMENTS,
      zip_safe=False,
      extras_require=extras)
