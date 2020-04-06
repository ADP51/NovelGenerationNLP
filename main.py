from __future__ import absolute_import, division, print_function, unicode_literals
import author_model as am
import tensorflow as tf
import csv

import numpy as np
import os

EPOCHS = 100     # EPOCHS is the amount of times the model is trained
BATCH_SIZE = 128 # number of samples that will be propagated through the network
BUFFER_SIZE = 10000 # Buffer size to shuffle the dataset
seq_length = 100
embedding_dim = 300     # The embedding dimension
rnn_units = 1024        # Number of RNN units

def read_csv(path_to_file, key, key_header, value_header):
    text = ""
    with open(path_to_file, newline='') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=',')
        for row in spamreader:
            if key in row[key_header]:
                new_line = row[value_header].strip('\"')
                text += new_line
                text += " "

    if(len(text) < 1000000):
        print(f"Corpus size : {len(text)}, consider using larger dataset.")
    else:
        print(f"Corpus size : {len(text)}")

    return text

def read_text(path):
    # Read all file paths in corpora directory
    file_list = []
    for root, _ , files in os.walk(path): 
        for filename in files:
            file_list.append(os.path.join(root, filename))

    print("Read ", len(file_list), " files..." )

    # Extract text from all documents
    docs = []

    for files in file_list:
        with open(files, 'r') as fin:
            try:
                str_form = fin.read().lower().replace('\n', '')
                docs.append(str_form)
            except UnicodeDecodeError: 
                # Some sentences have wierd characters. Ignore them for now
                pass

    # Combine them all into a string of text
    text = ' '.join(docs)

    if(len(text) < 1000000):
        print(f"Corpus size : {len(text)}, consider using larger dataset.")
    else:
        print(f"Corpus size : {len(text)}")
    
    return text


def create_training_segments(segment):
    input_data = segment[:-1]
    training_data = segment[1:]
    return input_data, training_data


# set up loss for model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def read_corpus(text):
    # The unique characters in the file
    vocab = sorted(set(text))
    print(f"Vocabulary size : {len(vocab)}")

    return vocab



def start_model(char_idx, text):

    text_as_int = np.array([char_idx[c] for c in text])

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    segments = char_dataset.batch(seq_length+1, drop_remainder=True)

    # map the input / training segments
    dataset = segments.map(create_training_segments)

    # shuffle the dataset
    return dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def main():
    text = read_text("./data/text-clean/poe")
    vocab = read_corpus(text)

    char_idx = {u: i for i, u in enumerate(vocab)}
    idx_char = np.array(vocab)

    dataset = start_model(char_idx, text)

    # Length of the vocabulary
    vocab_size = len(vocab)

    # build the model
    model = am.build_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    model.compile(optimizer='adam', loss=loss)
    # model.load_weights(tf.train.latest_checkpoint("./training/poe"))

    # Directory where the checkpoints will be saved
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join("./training/poe", "ckpt_{epoch}")

    #callback function called at the end of epoch training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    am.save_char_mapping(vocab, 'poe_map.csv')
    # am.train_model(model, dataset, EPOCHS, checkpoint_callback)

    new_model = am.build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=1)

    new_model.load_weights(tf.train.latest_checkpoint("./training/poe"))
    new_model.build(tf.TensorShape([1, None]))
    new_model.summary()
    print(am.generate_text(new_model, "never more", char_idx, idx_char))


if __name__ == '__main__':
    main()
