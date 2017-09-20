from elit.tokenizer import english_tokenizer
import os
import unittest

RESOURCE_PATH = './../elit/resources/tokenizer'


def resource_file(filename):
    return "{0}/{1}".format(RESOURCE_PATH, filename)


class EmoticonsTest(unittest.TestCase):

    def setUp(self):
        # self.file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
        #                          resource_file('EMOTICONS'))
        pass

    def test_tokenizer(self):
        test_case = ':smile: :hug: :pencil:'
        tokens = english_tokenizer.tokenize(test_case)
        self.assertEqual(len(tokens), 3)


if __name__ == '__main__':
    unittest.main()
