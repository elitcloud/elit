# -*- coding:utf-8 -*-
# Filename: saveable.py
# Author：hankcs
# Date: 2018-02-28 12:44
import pickle


class Savable(object):
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
