# -*- coding: utf-8 -*-

import setuptools
import sys

with open("sklearnbot/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

dependency_links = []

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


setuptools.setup(name="sklearnbot",
                 author="Jan N. van Rijn",
                 description="Runs scikit-learn bot on classifiers and uploads results to OpenML.",
                 license="BSD 3-clause",
                 url="https://www.openml.org/",
                 version=version,
                 packages=setuptools.find_packages(),
                 package_data={'': ['*.txt', '*.md']},
                 install_requires=[
                     'ConfigSpace',
                     'openml',
                     'openmlcontrib',
                     'pandas>=0.24.0',
                 ],
                 test_suite="pytest",
                 classifiers=['Intended Audience :: Science/Research',
                              'Intended Audience :: Developers',
                              'License :: OSI Approved :: BSD License',
                              'Programming Language :: Python',
                              'Topic :: Software Development',
                              'Topic :: Scientific/Engineering',
                              'Operating System :: POSIX',
                              'Operating System :: Unix',
                              'Operating System :: MacOS',
                              'Programming Language :: Python :: 3.6',
                              'Programming Language :: Python :: 3.7'])
