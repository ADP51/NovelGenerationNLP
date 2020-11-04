import re
import pandas as pd
import numpy as np
from time import time
from collections import defaultdict
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file

import spacy
import logging
import sys

EPOCHS = 100     # EPOCHS is the amount of times the model is trained
BATCH_SIZE = 128  # number of samples that will be propagated through the network
BUFFER_SIZE = 10000  # Buffer size to shuffle the dataset
seq_length = 100
embedding_dim = 300     # The embedding dimension
rnn_units = 1024
max_sentence_len = 40

logging.basicConfig(stream=sys.stdout, format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt='%H:%M:%S', level=logging.INFO)


def cleaning(doc):
    # Begin lemmatization
    txt = [token.lemma_ for token in doc]

    # Remove any sentence less than 2 words
    if len(txt) > 2:
        return ' '.join(txt)


def word2idx(word, word_model):
    return word_model.wv.vocab[word].index

  
def idx2word(idx, word_model):
  return word_model.wv.index2word[idx]


t = time()

print("reading data...")
df = pd.read_csv('./data/simpsons_dataset.csv')

max_sentence_len = df['spoken_words'].str.len().max().astype(np.int32)
print(f"Max sentence length: {max_sentence_len}\n")

df.isnull().sum()

df = df.dropna().reset_index(drop=True)
print(df.isnull().sum())

nlp = spacy.load('en', disable=['ner', 'parser'])

# Remove special char's
quick_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower()
                for row in df['spoken_words'])

# https://spacy.io/usage/spacy-101#pipelines
txt = [cleaning(doc) for doc in nlp.pipe(
    quick_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean: {} min'.format(round((time() - t)/60, 2)))

df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()

# https://radimrehurek.com/gensim/models/phrases.html
# Automatically detect common phrases – aka multi-word expressions, word n-gram collocations – from a stream of sentences.
print(df_clean['clean'])
sent = [row.split() for row in df['spoken_words']]
print(sent)

phrases = Phrases(sent, min_count=30, progress_per=10000)

bigram = Phraser(phrases)

sentences = bigram[sent]

w2v_model = Word2Vec(min_count=20, window=4, size=300, sample=6e-5,
                    alpha=0.03, min_alpha=0.0007, negative=20, workers=2)

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.train(sentences, total_examples=w2v_model.corpus_count,
                epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.wv.save_word2vec_format('./saved_models/model.bin', binary=True)
w2v_model.init_sims(replace=True)
pretrained_weights = w2v_model.wv.syn0
vocab_size, embedding_size = pretrained_weights.shape

print("Preparing Training data...")
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence[:-1]):
        train_x[i,t] = word2idx(word, w2v_model)
    print(f"Debug: {sentence[-1]}")
    train_y[i] = word2idx(sentence[-1], w2v_model)

print('\nBuilding Model...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))
model.add(GRU(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word, w2v_model) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx, w2v_model) for idx in word_idxs)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    'stupid flanders ',
    'I am so smart '
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))

model.fit(train_x, train_y,
          batch_size=128,
          epochs=20,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
