#!/usr/bin/env python

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

config = {
    'name':'elit',
    'description':'The Emory Language Information Toolkit (ELIT).',
    'author':'Jinho D. Choi',
    'author_email':'choi@mathcs.emory.edu',
    'url':'https://github.com/emorynlp/elit',
    'download_url':'https://github.com/emorynlp/elit/tree/master',
    'version':'0.1.0',
    'install_requires':[
        'nose',
        'Cython>=0.25',
        'mxnet>=0.9.4',
        'argparse>=1.4',
        'gensim>=1',
        'fasttext>=0.8'
    ],
    'packages':['elit'],
    'scripts':[],
    'license':'ALv2',
    'classifiers':[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
    ]
}

setup(**config)