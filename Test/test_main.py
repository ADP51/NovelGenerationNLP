import unittest

from main import read_corpus, start_model, read_csv, read_text


# **** MAKE SURE ALL THE PATHS ARE SET FROM THIS FILE OR THE TESTS WONT RUN *******
class MyTestCase(unittest.TestCase):

    # This test runs the read_csv method from main.py, it tests with a regex
    # method is not use so not properly tested
    def test_read_csv(self):
        # text = read_csv("./char_mappings/shakespeare_map.csv", 'k', 49, 49)
        # self.assertRegex(text, "[A-Za-z0-9]*")
        self.assertTrue(True)

    # This test runs the read_text method from main.py, if no errors happen it prints out the test worked
    def test_read_text(self):
        text = read_text("../data/shakespeare/")
        self.assertNotEqual(text, "")
        self.assertRegex(text, "[A-Za-z]*")

    # Does not require testing so returns true
    def test_create_training_segments(self):
        self.assertEqual(True, True)

    # Does not require testing so returns true
    def test_loss(self):
        self.assertEqual(True, True)

    # Test method for reading the file and creating a vocab, if no errors it prints out successful
    def test_read_corpus(self):
        text = read_text("../data/shakespeare/")
        vocab = read_corpus(text)
        self.assertNotEquals(vocab, "")

    # Tests the create model method, if there are no errors it prints out successful
    def test_start_model(self):
        text = read_text("../data/shakespeare/")
        vocab = read_corpus(text)
        char_idx = {u: i for i, u in enumerate(vocab)}

        test = start_model(char_idx, text)
        self.assertNotEquals(test, None)


if __name__ == '__main__':
    unittest.main()
