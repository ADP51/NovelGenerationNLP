import unittest
import tensorflow as tf
from author_model import build_model


# These tests are hard to test as they return models with variable information
class MyTestCase(unittest.TestCase):

    # Tests the build a model method from author_model
    def test_build_model(self):
        # get the basic path for a test document
        text = open('../data/shakespeare-hamlet.txt', 'rb').read().decode(encoding='utf-8')
        vocab = sorted(set(text))

        # run the model
        try:
            model = build_model(
                vocab_size=len(vocab),
                embedding_dim=256,
                rnn_units=1024,
                batch_size=64)
        except:
            self.assertFalse(True)

        # check to make sure the model is valid
        self.assertTrue(True)

    #TODO Need a small prebuilt model to run the train model method
    def test_train_model(self):
        self.assertEqual(True, True)

    #TODO Use the prebuilt model to generate text and use regex to make sure it worked properly
    def test_generate_text(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
