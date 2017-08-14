from elit.tokenizer.src import tokenizer
import unittest


class TokenizerTest(unittest.TestCase):
    def test_tokenizer(self):
        # print(tokenizer.tokenize("world"))
        # assert tokenizer.tokenize("world") == "tokenizer, world"
        print(tokenizer.vectorize("hello world"))


if __name__ == '__main__':
    unittest.main()
