#!/usr/bin/env python

from __future__ import division, print_function

import sys

from setuptools import setup, find_packages

from util import write_version_py, get_version_info

VERSION_PY = 'elit/version.py'

if sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version >= 3.4 required.")

# How mature is this project? Common values are
#   3 - Alpha
#   4 - Beta
#   5 - Production/Stable
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: Apache Software License
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Information Analysis
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

EXCLUDE_FROM_PACKAGES = ['']


def setup_package():
    write_version_py(filename=VERSION_PY)

    metadata = dict(
        name='elit',
        url='https://github.com/elitcloud/elit',
        download_url='https://github.com/elitcloud/elit/tree/master',
        author='Jinho D. Choi',
        author_email='jinho.choi@emory.edu',
        description='The Emory Language Information Toolkit (ELIT).',
        license='ALv2',
        packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
        package_data={'': ['*.txt']},
        install_requires=[
            'argparse',
            'pybind11',
            'yafasttext',
            'gensim',
            'numpy',
            'keras',
            'tensorflow',
            'ujson',
            'mxnet',
        ],
        tests_require=[
            'pytest',
        ],
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        scripts=['bin/elit'],
    )
    metadata['version'] = get_version_info(filename=VERSION_PY)[0]

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
