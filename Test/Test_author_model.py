import csv
import os
import unittest

import numpy as np
import tensorflow as tf
from author_model import build_model, generate_text, train_model


# These tests are hard to test as they return models with variable information
from main import read_shakespeare, read_corpus, start_model, loss


class MyTestCase(unittest.TestCase):

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

    # TODO Need to replace the None values with the correct model
    def test_train_model(self):

        text = read_shakespeare()
        vocab = read_corpus(text)

        # build model
        model = build_model(
            vocab_size=len(vocab),
            embedding_dim=256,
            rnn_units=1024,
            batch_size=64)

        model.compile(optimizer='adam', loss=loss)
        model.load_weights(tf.train.latest_checkpoint("./testing/shakespeare"))

        char_idx = {u: i for i, u in enumerate(vocab)}
        dataset = start_model(char_idx, text)

        epochs = 1

        checkpoint_prefix = os.path.join("./testing/shakespeare", "ckpt_{epoch}")

        # callback function called at the end of epoch training
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        # run the train_model method
        new_model = train_model(model, dataset, epochs, checkpoint_callback)
        self.assertTrue(new_model.built)

    # TODO Need to replace the None values with the correct model
    def test_generate_text(self):
        id_to_char = []
        with open("./char_mappings/shakespeare_map.csv") as file:
            reader = csv.reader(file)
            for row in reader:
                id_to_char.append(row[1])
            char_to_id = {k: v for v, k in enumerate(id_to_char)}

        model = load_model(len(char_to_id), '../training/shakespeare')

        output = generate_text(model, 'This is a test string', char_to_id, id_to_char)

        # run a Regex check to make sure it prints out correctly
        self.assertRegex(output, "This is a test string .*")


# Taken from the flask API repository
def load_model(vocab_size, author_ckpt_path):

    model = build(vocab_size, 256)
    model.load_weights(tf.train.latest_checkpoint(author_ckpt_path))
    model.build(tf.TensorShape([1, None]))
    return model


# Taken from the flask API repository
def build(vocab_size, shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, shape,
                                  batch_input_shape=[1, None]),
        tf.keras.layers.GRU(1024,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


if __name__ == '__main__':
    unittest.main()
