from elit.tokenizer.src import tokenizer
import unittest


class TokenizerTest(unittest.TestCase):
    def test_tokenizer(self):
        assert tokenizer.tokenize("world") == "tokenizer, world"


if __name__ == '__main__':
    unittest.main()
