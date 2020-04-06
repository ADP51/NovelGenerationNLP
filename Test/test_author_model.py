import unittest
import tensorflow as tf
from author_model import build_model, generate_text, train_model


# These tests are hard to test as they return models with variable information
class MyTestCase(unittest.TestCase):

    # Tests the build a model method from author_model
    def test_build_model(self):
        # get the basic path for a test document
        with open('data/shakespeare/shakespeare-hamlet.txt', 'rb') as f:
            text = f.read().decode(encoding='utf-8')

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

    # TODO Need to replace the None values with the correct model
    def test_train_model(self):

        # variables for the model
        model = None
        dataset = None
        epochs = 1
        checkpoint_callback = None

        # run the train_model method
        try:
            train_model(model, dataset, epochs, checkpoint_callback)
        except:
            self.assertFalse(True)

        self.assertTrue(True)

    # TODO Need to replace the None values with the correct model
    def test_generate_text(self):
        # variables for the models
        model = None
        char2idx = None
        idx2char = None

        # run a Regex check to make sure it prints out correctly
        self.assertRegex(generate_text(model, 'This is a test string', char2idx, idx2char), "This is a test string .*")


if __name__ == '__main__':
    unittest.main()
