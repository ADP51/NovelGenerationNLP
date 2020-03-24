from __future__ import absolute_import, division, print_function, unicode_literals
from author_model import build_model, train_model, generate_text
import tensorflow as tf

import numpy as np
import os

EPOCHS = 30     # EPOCHS is the amount of times the model is trained
BATCH_SIZE = 64
BUFFER_SIZE = 10000     # Buffer size to shuffle the dataset
seq_length = 100
embedding_dim = 256     # The embedding dimension
rnn_units = 1024        # Number of RNN units


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def open_file():
    # import shakespeare text files
    path_to_file = tf.keras.utils.get_file(
        'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    return open(path_to_file, 'rb').read().decode(encoding='utf-8')


def read_file(text):
    # length of text is the number of characters in it
    print('Length of raw text: {} characters'.format(len(text)))

    # The unique characters in the file
    vocab = sorted(set(text))
    print('Vocabulary size: {}'.format(len(vocab)))

    return vocab


def start_model(char2idx, text):

    text_as_int = np.array([char2idx[c] for c in text])

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    # map the input / target sequences
    dataset = sequences.map(split_input_target)

    # shuffle the dataset
    return dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def main():
    text = open_file()
    vocab = read_file(text)

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    dataset = start_model(char2idx, text)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    # build the model
    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    model = train_model(model, dataset, EPOCHS, checkpoint_callback)

    # TODO finish this code, cause model has to print something and this is not it
    generate_text(model, "This is where the story begins", char2idx, idx2char)


if __name__ == '__main__':
    main()
