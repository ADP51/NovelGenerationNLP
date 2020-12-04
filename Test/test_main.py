import unittest
import tensorflow as tf

from main import read_corpus, start_model, read_text, read_csv


# **** MAKE SURE ALL THE PATHS ARE SET FROM THIS FILE OR THE TESTS WONT RUN *******
class TestMain(unittest.TestCase):

    # This test runs the read_text method from main.py, if no errors happen it prints out the test worked
    def test_read_text(self):
        text = read_text("../data/shakespeare/")
        self.assertNotEqual(text, "")
        self.assertRegex(text, "[A-Za-z]*")

    # Test method for reading the file and creating a vocab, if no errors it prints out successful
    def test_read_corpus(self):
        text = read_text("../data/shakespeare/")
        vocab = read_corpus(text)
        self.assertNotEqual(vocab, "")

    # Tests the create model method, if there are no errors it prints out successful
    def test_start_model(self):
        text = read_text("../data/shakespeare/")
        vocab = read_corpus(text)
        char_idx = {u: i for i, u in enumerate(vocab)}

        test = start_model(char_idx, text)
        self.assertNotEqual(test, None)


if __name__ == '__main__':
    unittest.main()
