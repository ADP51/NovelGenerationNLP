import tensorflow as tf

"""
Using a fixed max sequence length can increase the training speed and also 
help the language model avoid overfitting uncommon text dependencies 
(which are sometimes found in long, run-on sentences).
"""
def truncate_sequences(sequence, max_length):
    input_sequence = sequence[:max_length-1]
    target_sequence = sequence[1:max_length]
    return input_sequence, target_sequence

# For each sequence that's shorter than the maximum sequence length, 
# we append a special non-vocabulary token to the end of the sequence 
# until its length is equal to the maximum sequence length.
def pad_sequences(sequence, max_length):
    padding_amount = max_length - len(sequence)
    padding = [0 for i in range(padding_amount)]
    input_sequence = sequence[:-1] + padding
    target_sequence = sequence[1:] + padding
    return input_sequence, target_sequence

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def get_input_target_sequence(self, sequence):
        seq_len = len(sequence)
        if seq_len >= self.max_length:
            input_sequence, target_sequence = truncate_sequences(
                sequence, self.max_length
            )
        else:
            # Next chapter
            input_sequence, target_sequence = pad_sequences(
                sequence, self.max_length
            )
        return input_sequence, target_sequence
 
    #create the lstm cells     
    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units) 
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        return dropout_cell

    def stacked_lstm_cells(self, is_training):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob) for i in range(self.num_lstm_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        return cell

   def get_input_embeddings(self, input_sequences):
        embedding_dim = int(self.vocab_size**0.25)
        initial_bounds = 0.5 / embedding_dim
        initializer = tf.random_uniform(
            [self.vocab_size, embedding_dim],
            minval=-initial_bounds,
            maxval=initial_bounds)
        self.input_embedding_matrix = tf.get_variable('input_embedding_matrix',
            initializer=initializer)
        input_embeddings = tf.nn.embedding_lookup(self.input_embedding_matrix, input_sequences)
        return input_embeddings
    
    # Run the LSTM on the input sequences
    def run_lstm(self, input_sequences, is_training):
        cell = self.stacked_lstm_cells(is_training)
        input_embeddings = self.get_input_embeddings(input_sequences)
        binary_sequences = tf.sign(input_sequences)
        sequence_lengths = tf.reduce_sum(binary_sequences, axis=1)
        lstm_outputs, _ = tf.nn.dynamic_rnn(
        cell,
        input_embeddings,
        sequence_length=sequence_lengths,
        dtype=tf.float32)
        return lstm_outputs, binary_sequences 






