import re
import random
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

for i in range(10):
  print(df_clean.sample())
  print(df.sample())

# https://radimrehurek.com/gensim/models/phrases.html
# Automatically detect common phrases – aka multi-word expressions, word n-gram collocations – from a stream of sentences.
sent = [row.split() for row in df['spoken_words']]

phrases = Phrases(sent, min_count=30, progress_per=10000)

bigram = Phraser(phrases)

sentences = bigram[sent]
print(type(sentences))

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

# lastWord = 'beer'
# genTxt = []

# for i in range(500):
#   extract = random.choice(w2v_model.wv.most_similar(positive=[lastWord]))
#   lastWord = extract[0]
#   print(lastWord)
#   genTxt.append(lastWord)

sentence2Change = df['spoken_words'].sample()
new_sentence = [w2v_model.wv.most_similar(positive=[word])[0][0] for word in sentence2Change]
print(' '.join(new_sentence))

