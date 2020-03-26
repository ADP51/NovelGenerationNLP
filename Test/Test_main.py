import unittest
from main import open_file, read_file, compile_model, load_model
import tensorflow as tf


class MyTestCase(unittest.TestCase):

    def test_open_file(self):
        text = open_file()
        self.assertEqual(text, open(tf.keras.utils.get_file(
            'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'),
            'rb').read().decode(encoding='utf-8'))

    def test_read_file(self):
        text = open_file()
        vocab = read_file(text)
        self.assertEqual(vocab, sorted(set(text)))

    # TODO write this test
    def test_start_model(self):
        self.assertEqual(True, True)

    def test_compile_model(self):
        text = open_file()
        vocab = read_file(text)

        try:
            model = compile_model(len(vocab))
        except:
            self.assertFalse(True)

        self.assertEqual(True, True)

    # TODO Not sure how to test this as it only gets called and has no model within it
    def test_save_model(self):
        self.assertEqual(True, True)

    # TODO Doesn't work yet
    def test_load_model(self):
        text = open_file()
        vocab = read_file(text)
        model = compile_model(len(vocab))

        try:
            model = load_model(model)
        except:
            self.assertFalse(True)
        self.assertEqual(True, True)

    # TODO Write the test
    def test_char_to_idx(self):
        self.assertEqual(True, True)

    # TODO Write the test
    def test_idx_to_char(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
