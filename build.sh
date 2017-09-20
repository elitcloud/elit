#!/usr/bin/env sh

rm -rf build
python3 setup.py build
python3 setup.py build_ext --inplace
