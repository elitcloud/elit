#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from subprocess import call
import os

EXCLUDE_FROM_PACKAGES = ['']

BASEPATH = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_BUILD_PATH = os.path.join(BASEPATH, 'elit/tokenizer/src')


class TokenizerInstall(install):
    """
    Class for build the C library of tokenizer
    """

    def run(self):
        """
        Build and install the shared library of hostapd
        """

        def compile_tokenizer():
            """
            Compile the shared library of hostapd
            """
            call(['make'], cwd=TOKENIZER_BUILD_PATH)

        # Before installing, we compile the tokenizer library first
        self.execute(compile_tokenizer, [],
                     'Compiling tokenizer library')

        install.run(self)


extension_mod = Extension("_tokenizer",
                          ["./elit/tokenizer/src/_tokenizer_module.cc", "./elit/tokenizer/src/tokenizer.cc"])

setup(
    name='elit',
    version='0.1.0',
    url='https://github.com/emorynlp/elit',
    download_url='https://github.com/emorynlp/elit/tree/master',
    author='Jinho D. Choi',
    author_email='choi@mathcs.emory.edu',
    description='The Emory Language Information Toolkit (ELIT).',
    license='ALv2',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=[
        'nose',
        'Cython',
        'mxnet',
        'argparse',
        'gensim',
        'fasttext',
        'numpy'
    ],
    classifiers=[
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
    ],
    ext_modules=[extension_mod],
    cmdclass={
        'install': TokenizerInstall,
    }
)
