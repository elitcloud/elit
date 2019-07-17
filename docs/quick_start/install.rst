Installation
============

-----------
GPU Machine
-----------

Deep learning models in ELIT require a GPU machine.
Take a look at the `pre-trained models <../documentation/models.html>`_ page to check how much GPU memories are needed for models of your choice.
If you do not have a GPU machine powerful enough to run those models, you may want to consider using our `web APIs <decode_webapi.html>`_.

----------------
CUDA Environment
----------------

ELIT requires the following `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ environment:

* Supported versions of the CUDA Toolkit: `8.0 <https://developer.nvidia.com/cuda-80-download-archive>`_, `9.0 <https://developer.nvidia.com/cuda-90-download-archive>`_, `9.2 <https://developer.nvidia.com/cuda-92-download-archive>`_, `10.0 <https://developer.nvidia.com/cuda-10.0-download-archive>`_, `10.1 <https://developer.nvidia.com/cuda-10.1-download-archive>`_.
* CUDA Deep Neural Network library: `cuDNN <https://developer.nvidia.com/cudnn>`_.

The followings show how to install the `CUDA Toolkit 10.1 <https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork>`_ on Ubuntu 18.04:

.. code:: console

   $ wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
   $ sudo dpkg -i cuda-repo-ubuntu1804_10.1.168-1_amd64.deb
   $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   $ sudo apt update
   $ sudo apt install cuda
   $ nvidia-smi
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
   |-------------------------------+----------------------+----------------------+

The followings show how to install the `cuDNN 7.6.0.64 <https://developer.nvidia.com/rdp/cudnn-download>`_ on Ubuntu 18.04:

.. code:: console

   $ wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.6.0.64-1+cuda10.1_amd64.deb
   $ sudo dpkg -i libcudnn7_7.6.0.64-1+cuda10.1_amd64.deb
   $ sudo apt install libcudnn7
   libcudnn7 is already the newest version (7.6.0.64-1+cuda10.1).  

------------------
Python Environment
------------------

ELIT requires `Python <https://www.python.org/downloads/>`_ ≥ 3.6.
The followings show how to install the `Python 3.6.8 <https://www.python.org/downloads/release/python-368/>`_ on Ubuntu 18.04:

.. code-block:: console
   
   $ sudo apt install python3.6
   $ sudo apt install python3.6-dev
   $ sudo apt install python3-setuptools
   $ sudo apt install python3-pip
   $ sudo apt install python-virtualenv
   $ python3 --version
   Python 3.6.8

-------------------
Virtual Environment
-------------------

We recommend to install ELIT using `Virtualenv <https://virtualenv.pypa.io>`_.
The followings show how to setup a virtualenv using Python 3.6:

.. code-block:: console

   $ virtualenv --python=/usr/bin/python3.6 ~/.elit
   $ source ~/.elit/bin/activate
   (.elit) $

------------------
MXNet Installation
------------------

ELIT uses `Apache MXNet <https://mxnet.incubator.apache.org>`_ to develop deep learning models.
The followings show how to install `MXNet 1.4.1 <https://mxnet.incubator.apache.org/versions/master/install/index.html?platform=Linux&language=Python&processor=GPU>`_ based on `CUDA Toolkit 10.1 <https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork>`_:

.. code-block:: console

   (.elit) $ pip install mxnet-cu101
   (.elit) $ pip show mxnet-cu101
   Name: mxnet-cu101
   Version: 1.4.1

------------------
ELIT Installation
------------------

Finally, the followings show how to install the latest version of ELIT:

.. code-block:: console

   (.elit) $ pip install elit
   Name: elit
   Version: 0.2.0
