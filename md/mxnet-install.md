# MXNet Installation on AWS

## Launch an Amazon Machine Image (AMI)

* Step 1: Ubuntu Server 16.04 LTS (HVM), SSD Volume Type.
* Step 2: GPU compute, `p2.xlarge`.
* Step 3: default.
* Step 4: Increase the size to `80 GiB` (or more).
* Step 5: default.
* Step 6: default.

## Login to AMI

```
ssh -i my_key_pair.pem ubuntu@ec2-xxx-xxx-xxx-xxx.compute-x.amazonaws.com
```

## CUDA 8.x

Source: https://developer.nvidia.com/cuda-toolkit

```
CUDA=cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/$CUDA
sudo dpkg -i $CUDA
sudo apt-get update
sudo apt-get install cuda
```

## cuDNN 5.x

Source: https://developer.nvidia.com/cudnn

```
CUDNN=cudnn-8.0-linux-x64-v5.1.tgz
wget https://dl.dropboxusercontent.com/u/15060914/opt/$CUDNN
tar xvzf $CUDNN
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig
```

## Python 3.6

```
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3.6-dev
sudo apt-get install python3-setuptools
```

## Virtual Environment

```
sudo apt-get install python-pip python-dev python-virtualenv
virtualenv --python=/usr/bin/python3.6 ~/.mxnet
source ~/.mxnet/bin/activate
```

## OpenBLAS

```
sudo apt-get install libopenblas-dev
```

## OpenCV

Source: https://www.hiroom2.com/2016/05/20/ubuntu-16-04-install-opencv-3-1/

```
sudo apt-get build-dep -y opencv
echo 'LD_LIBRARY_PATH=/usr/local/lib' >> .bashrc
git clone https://github.com/Itseez/opencv
cd opencv
git checkout 3.1.0 -b 3.1.0
cmake -G 'Unix Makefiles' \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -DCMAKE_BUILD_TYPE=Release  \
  -DBUILD_EXAMPLES=ON \
  -DINSTALL_C_EXAMPLES=ON \
  -DINSTALL_PYTHON_EXAMPLES=ON  \
  -DBUILD_NEW_PYTHON_SUPPORT=ON \
  -DWITH_FFMPEG=ON  \
  -DWITH_GSTREAMER=OFF  \
  -DWITH_GTK=ON \
  -DWITH_JASPER=ON  \
  -DWITH_JPEG=ON  \
  -DWITH_PNG=ON \
  -DWITH_TIFF=ON  \
  -DWITH_OPENEXR=ON \
  -DWITH_PVAPI=ON \
  -DWITH_UNICAP=OFF \
  -DWITH_EIGEN=ON \
  -DWITH_XINE=OFF \
  -DBUILD_TESTS=OFF \
  -DCMAKE_SKIP_RPATH=ON \
  -DWITH_CUDA=OFF \
  -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DENABLE_SSE=ON -DENABLE_SSE2=ON -DENABLE_SSE3=OFF \
  -DWITH_OPENGL=ON -DWITH_TBB=ON -DWITH_1394=ON -DWITH_V4L=ON
make && sudo make install
cd ..
```

## Numpy

```
pip3 install numpy
```

## GCC

Source: https://github.com/dmlc/mxnet/issues/4978

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50 --slave /usr/bin/g++ g++ /usr/bin/g++-5
```

## MXNet

Source: http://mxnet.io/get_started/ubuntu_setup.html#build-the-shared-library

```
sudo apt-get update
sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev
git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive
cd ~/mxnet
cp make/config.mk .
echo "USE_BLAS=openblas" >> config.mk
echo "ADD_CFLAGS += -I/usr/include/openblas" >> config.mk
echo "ADD_LDFLAGS += -lopencv_core -lopencv_imgproc" >> config.mk
echo "USE_CUDA=1" >> config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk
echo "USE_CUDNN=1" >> config.mk
make -j$(nproc)
```

If you see an error like this:

```
/usr/bin/ld: cannot find -lippicv
```

Try the following:

```
cmake -DWITH_IPP=ON -DINSTALL_CREATE_DISTRIB=ON .
make -j$(nproc)
```

Finally, export the `mxnet` package:

```
cp libmxnet.so python/mxnet/
echo 'export PYTHONPATH=$PYTHONPATH:/home/ubuntu/mxnet/python' >> ~/.bashrc
```