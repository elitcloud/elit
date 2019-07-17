# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-24 16:16
import importlib

from elit.component.tagger.corpus import Dictionary


def type_to_str(type_object) -> str:
    """
    convert a type object to class path in str format
    :param type_object: type
    :return: class path
    """
    cls_name = str(type_object)
    assert cls_name.startswith("<class '"), 'illegal input'
    cls_name = cls_name[len("<class '"):]
    assert cls_name.endswith("'>"), 'illegal input'
    cls_name = cls_name[:-len("'>")]
    return cls_name


def class_path_of(obj) -> str:
    """
    get the full class path of object
    :param obj:
    :return:
    """
    return "{0}.{1}".format(obj.__class__.__module__, obj.__class__.__name__)


def str_to_type(classpath) -> type:
    """
    convert class path in str format to a type
    :param classpath: class path
    :return: type
    """
    module_name, class_name = classpath.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls


if __name__ == '__main__':
    cls_path = type_to_str(type(Dictionary()))
    print(cls_path)
    print(str_to_type(cls_path))
    print(class_path_of(Dictionary()))
