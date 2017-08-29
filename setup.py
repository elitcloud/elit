#!/usr/bin/env python

from __future__ import division, print_function
import os
import sys
import subprocess

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
Programming Language :: C
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

MAJOR = 0
MINOR = 1
MICRO = 2
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('elit/version.py'):
        # must be a source distribution, use existing version file
        try:
            from elit.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing " \
                              "elit/version.py and the build directory " \
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='elit/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SCIPY SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


EXCLUDE_FROM_PACKAGES = ['']


def setup_package():
    from setuptools import setup, find_packages, Extension
    from setuptools.command.install import install
    from subprocess import call

    BASEPATH = os.path.dirname(os.path.abspath(__file__))
    TOKENIZER_BUILD_PATH = os.path.join(BASEPATH, 'elit/tokenizer')

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
                              ["./elit/tokenizer/_tokenizer_module.cpp",
                               "./elit/tokenizer/tokenizer.cpp"],
                              language='c++',
                              extra_compile_args=['-std=c++0x', '-std=c++11'])

    metadata = dict(
        name='elit',
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
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        ext_modules=[extension_mod],
        cmdclass={
            'install': TokenizerInstall,
        }
    )
    metadata['version'] = get_version_info()[0]

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
