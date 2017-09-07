# Installing the ELIT

## Prerequisites 

- python >= 3.4
- gcc >= 5
- boost >= 1.55

### Ubuntu & Debian

```commandline
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-5 g++-5 libboost-all-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
```

### Mac
```commandline
brew install boost
```

## Install via pip

```commandline
pip install Cython
pip install elit
```