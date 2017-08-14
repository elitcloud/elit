from elit.tokenizer import tokenizer
import os
import unittest

RESOURCE_PATH = './../resources/tokenizer'


def resource_file(filename):
    return "{0}/{1}".format(RESOURCE_PATH, filename)


class EmoticonsTest(unittest.TestCase):

    def setUp(self):
        self.file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 resource_file('EMOTICONS'))

    def test_tokenizer(self):
        with open(self.file) as f:
            lines = f.readlines()
        for line in lines:
            test_case = line.rstrip('\n')
            with self.subTest(line=test_case):
                self.assertEqual(sorted(tokenizer.tokenize(test_case)), sorted(tuple(test_case.split(" "))))


if __name__ == '__main__':
    unittest.main()
