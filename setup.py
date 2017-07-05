try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'The Emory Language Information Toolkit (ELIT).',
    'author': 'Jinho D. Choi',
    'url': 'https://github.com/emorynlp/elit',
    'download_url': 'git@github.com:emorynlp/elit.git',
    'author_email': 'choi@mathcs.emory.edu',
    'version': '0.1',
    'install_requires': [
        'nose',
        'Cython>=0.25',
        'mxnet>=0.9.4',
        'argparse>=1.4',
        'gensim>=1',
        'fasttext>=0.8'
        ],
    'packages': ['elit'],
    'scripts': [],
    'name': 'elit'
}

setup(**config)