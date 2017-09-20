#!/usr/bin/env python

from __future__ import division, print_function
import os
import sys
import subprocess

from setuptools import setup, Extension
import setuptools
from setuptools.command.build_ext import build_ext

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
MICRO = 6
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


class PybindInclude(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


def setup_package():
    write_version_py()

    ext_modules = [
        Extension(
            'elit.tokenizer.english_tokenizer',
            ['elit/cpp/tokenizer/english_tokenizer_py.cpp',
             'elit/cpp/tokenizer/english_tokenizer.cpp',
             'elit/cpp/tokenizer/token_utils.cpp',
             'elit/cpp/string_utils.cpp',
             'elit/cpp/io_utils.cpp'],
            include_dirs=[
                # Path to pybind11 headers
                PybindInclude(),
                PybindInclude(user=True),
            ],
            language='c++'
        ),
    ]

    metadata = dict(
        name='elit',
        url='https://github.com/emorynlp/elit',
        download_url='https://github.com/emorynlp/elit/tree/master',
        author='Jinho D. Choi',
        author_email='choi@mathcs.emory.edu',
        description='The Emory Language Information Toolkit (ELIT).',
        license='ALv2',
        packages=['elit'],
        package_data={'': ['resources/tokenizer/*.txt',
                           'resources/sample/*']},
        include_package_data=True,
        install_requires=[
            'Cython',
            'mxnet',
            'argparse',
            'gensim',
            'fasttext',
            'numpy',
            'pybind11>=2.2'
        ],
        tests_require=[
            'nose',
        ],
        test_suite='nose.collector',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExt}
    )
    metadata['version'] = get_version_info()[0]

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
