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
    'install_requires': ['nose'],
    'packages': ['elit'],
    'scripts': [],
    'name': 'elit'
}

setup(**config)