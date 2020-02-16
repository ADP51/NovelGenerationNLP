
import nltk
import numpy as np
import os
import random
import sys

from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop

#file path for all of the state of the union addresses
corpora_dir = "/Users/andrewpalmer/nltk_data/corpora/state_union"

# Read all file paths in corpora directory
file_list = []
for root, _ , files in os.walk(corpora_dir):  
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

print('corpus length:', len(text))

# create a vocabulary of chars
chars = sorted(list(set(text)))
print('Total Number of Unique Characters:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars)) # Character to index
indices_char = dict((i, c) for i, c in enumerate(chars)) # Index to Character

"""
Recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
"""

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40 # Number of characters considered
step = 3 # Stide of our window
sentences = []
next_chars = []

# Rading the text in terms of sequence of characters
# Extract only 'maxlen' characters every time
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    # The character just after the sequence is the label
    next_chars.append(text[i + maxlen]) 
print('nb sequences:', len(sentences))

print('Vectorization...')
# Initializing Tensor (training data)
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool) 
# Initializing Output that holds next character (label)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) 
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        # Populate Tensor Input
        x[i, t, char_indices[char]] = 1 
    # Populate y with the character just after the sequence
    y[i, char_indices[next_chars[i]]] = 1


def sample(preds, temperature=1.0):
    """Perform Temperature Sampling"""
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature 
    exp_preds = np.exp(preds)
    # Softmax of predictions
    preds = exp_preds / np.sum(exp_preds) 
    # Sample a single characters, with probabilities defined in `preds`
    probas = np.random.multinomial(1, preds, 1) 
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    """Function invoked at end of each epoch. Prints generated text"""
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- Diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            # Generate next character
            next_index = sample(preds, diversity) 
            next_char = indices_char[next_index]
            
            # Append character to generated sequence
            generated += next_char 
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    
    # Save model weights into file
    model.save_weights('saved_weights.hdf5', overwrite=True)
        

# After every single epoch, we are going to call the function on_epoch_end
# to generate some text.
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

print('Building model...')
# Initialize Sequential Model
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
# Add the output layer that is a softmax of the number of characters
model.add(Dense(len(chars), activation='softmax')) 
# Optimization through RMSprop
optimizer_new = RMSprop() 
# Consider cross Entropy loss. Why? MLE of P(D | theta)
model.compile(loss='categorical_crossentropy', optimizer=optimizer_new) 

# Train this for 30 epochs. Size of output from LSTM i.e. hidden layer vector shape=128
model.fit(x, y,
          batch_size=128,
          epochs=30,
          callbacks=[print_callback, checkpointer])
