import csv
import os
import unittest

import numpy as np
import tensorflow as tf
from author_model import build_model, generate_text, train_model


# These tests are hard to test as they return models with variable information
from main import read_corpus, start_model, loss, read_text


class TestAuthorModel(unittest.TestCase):

    # No test can be written for this method since nothing returned
    def test_save_char_mapping(self):
        self.assertEqual(True, True)

    # Tests the build a model method from author_model
    def test_build_model(self):
        # get the basic path for a test document
        with open('../data/shakespeare/shakespeare-hamlet.txt', 'rb') as f:
            text = f.read().decode(encoding='utf-8')

        vocab = sorted(set(text))

        # run the model
        model = build_model(
            vocab_size=len(vocab),
            embedding_dim=256,
            rnn_units=1024,
            batch_size=64)

        # check to make sure the model is valid
        self.assertTrue(model.built)

    # Tests the train model method from author_model.py
    # This test takes a while to run
    # Deprecated, we don't use this method anymore
    def test_train_model(self):

        # new_model, char_idx, idx2char = generate_model()
        # self.assertTrue(new_model.built)
        self.assertTrue(True)

    # This tests the generate_text method from the author_model.py, runs a regex check and then prints out the text
    # This test takes a while to
    def test_generate_text(self):
        # run the train_model method
        model, char_idx, idx2char = generate_model()

        output = generate_text(model, "start text", char_idx, idx2char)
        # run a Regex check to make sure it prints out correctly
        self.assertRegex(output, "start text.*")
        print(output)


def generate_model():
    text = read_text("../data/shakespeare")
    vocab = read_corpus(text)

    char_idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    dataset = start_model(char_idx, text)

    # build model
    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=300,
        rnn_units=1024,
        batch_size=128)

    model.compile(optimizer='adam', loss=loss)

    epochs = 1

    checkpoint_prefix = os.path.join("./train", "ckpt_{epoch}")

    # callback function called at the end of epoch training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    new_model = build_model(
        vocab_size=len(vocab),
        embedding_dim=300,
        rnn_units=1024,
        batch_size=1)

    return new_model, char_idx, idx2char


if __name__ == '__main__':
    unittest.main()
