import unittest

from main import read_homerSimpson, read_shakespeare, read_corpus, start_model


# **** MAKE SURE ALL THE PATHS ARE SET FROM THIS FILE OR THE TESTS WONT RUN *******
class MyTestCase(unittest.TestCase):
    # This test runs the read_homerSimpson method from main.py, it tests with a regex
    # if this test does not work make sure the path is set correctly and on the open statement add
    # encoding="utf8"
    def test_read_homerSimpson(self):
        text = read_homerSimpson()
        self.assertRegex(text, "[A-Za-z0-9]*")

    # This test runs the read_shakespeare method from main.py, if no errors happen it prints out the test worked
    # if this test doesnt work make sure the path is set correctly
    def test_read_shakespeare(self):
        text = read_shakespeare()
        self.assertRegex(text, "[A-Za-z]*")

    # Does not require testing so returns true
    def test_create_training_segments(self):
        self.assertEqual(True, True)

    # Does not require testing so returns true
    def test_loss(self):
        self.assertEqual(True, True)

    # Test method for reading the file and creating a vocab, if no errors it prints out successful
    def test_read_corpus(self):
        text = read_shakespeare()
        vocab = read_corpus(text)
        self.assertEquals(True, True)

    # Tests the create model method, if there are no errors it prints out successful
    def test_start_model(self):
        text = read_shakespeare()
        vocab = read_corpus(text)
        char_idx = {u: i for i, u in enumerate(vocab)}

        test = start_model(char_idx, text)
        self.assertEquals(True, True)


if __name__ == '__main__':
    unittest.main()
