# The ELIT Project [![Build Status](https://travis-ci.org/elitcloud/elit.svg?branch=master)](https://travis-ci.org/elitcloud/elit) [![PyPI version](https://badge.fury.io/py/elit.svg)](https://badge.fury.io/py/elit) [![Documentation Status](https://readthedocs.org/projects/elit/badge/?version=latest)](http://elit.readthedocs.io/en/latest/?badge=latest)


The **E**mory **L**anguage **I**nformation **T**oolkit or **E**volution of **L**anguage and **I**nformation **T**echnology (ELIT) project provides:

* NLP tools readily available for research in various disciplines.
* Frameworks for fast development of efficient and robust NLP components.

The project is initiated and currently led by the [Emory NLP](http://nlp.mathcs.emory.edu) research group. It is under the [Apache 2](http://www.apache.org/licenses/LICENSE-2.0) license. Please join our group to get notifications about updates and send us your feedback.

## Installation

The machine should satisfy the requirements below before installing elit. If you installed the requirements below, you can skip next part of setup machine environment.  

- python > 3.4

### Setup machine environment

In this section, all the installation command execute

#### Ubuntu

```
sudo apt-get -y update
sudo apt-get -y install python3-pip python-dev build-essential
```

#### MacOS

On Mac OS, please install [homebrew](https://brew.sh/) first.

```
brew update
brew install python3
```


### Install Elit

There are many ways to start a python and install python package. To keep it simple, I use [virtualenv](https://github.com/pypa/virtualenv) to initialize a environment come with python 3 and use pip as my python package management tools.  

First of all, update your pip of python 3 to latest version:

```
pip3 install --upgrade pip
```

Create an virtualenv with python 3. `env` is your environment name, you can change it as you want. However, for simplicity, I use `env` in the rest of part. For much more usage, please check the [document](https://virtualenv.pypa.io/en/stable/userguide/).

```
virtualenv -p python3 env
```

Activate your virtualenv
```
source env/bin/activate
```

After you activate your virtualenv, you should your environment name in the starting of your command line, such as
```
(env) $
```

Let's assume your are running your python in the virtualenv, so I don't put the prefix `env` anymore.

Because of [fasttext](https://github.com/facebookresearch/fastText), cython are required and installed before we install elit. Since we're running python 3 in the virtualenv, we can just use pip instead of pip3.  

```
pip install cython
```

Now, we can install elit!

```
pip install elit
```

If you have any question or want to report bugs, please let us know on [github issues](https://github.com/elitcloud/elit/issues).

Thank you.
