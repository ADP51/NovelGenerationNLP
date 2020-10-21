import re
import pandas as pd
import pickle
from time import time
from itertools import islice
import string
from unidecode import unidecode
from collections import defaultdict
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import spacy
import logging
import sys

logging.basicConfig(stream=sys.stdout, format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

model_name = 'model'

t = time()

with open('./data/simpsons_dataset.txt', encoding='utf-8') as file:
    docs = file.readlines()

max_sentence_len = 40

# Translates non-ASCII characters, in case of bad input
docs_clean = [unidecode(line)[:max_sentence_len] for line in docs]


# Cleans and divides sentences into sentence arrays (arrays of arrays of words)
# sent_arr_raw = [[word for word in doc.lower().translate(str.maketrans('', '', string.punctuation))
#                                    .split()[:max_sentence_len]] for doc in docs]

print("Raw text: ")
for line in islice(docs, 0, 3):
    print(line)

# TODO: Investigate possible benefits of keeping punctuation, to get better spacy output
# Further cleaning, lower-casing and removal of punctuation
sent_arr_raw = [doc.lower().translate(str.maketrans('', '', string.punctuation)).rstrip() for doc in docs]

# TODO: Save this object and name it along with the model file, for use in w2v_gen

print("Cleaned samples: ")
for line in islice(sent_arr_raw, 0, 3):
    print(line)

# Ensuring correct model is loaded in
if spacy.util.is_package('en'):
    nlp = spacy.load('en', disable=['ner', 'parser'])

elif spacy.util.is_package('en_core_web_sm'):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])


def cleaning(doc):
    # Begin lemmatization and removing stopwords
    doc = nlp(doc)
    txt = [token.lemma_ for token in doc if not token.is_stop]

    # Remove any sentence less than 2 words
    if len(txt) > 2:
        return ' '.join(txt)


# https://spacy.io/usage/spacy-101#pipelines
sent_arr_lem = [cleaning(doc) for doc in sent_arr_raw]

print('Time to clean: {} min'.format(round((time()- t)/60,2)))
t = time()

print("Lemmatized samples: ")
for line in islice(sent_arr_lem, 0, 3):
    print(line)

sent_arr_split = [row.split() for row in sent_arr_lem if row]

phrases = Phrases(sent_arr_split, min_count=30, progress_per=10000)

bigram = Phraser(phrases)

# Trim ngram sentence length and remove empty lists
sent_arr_gram = [sentence[:max_sentence_len] for sentence in bigram[sent_arr_split] if sentence]

# TODO: Don't limit anything until this stage. Here, split any over-sized sentence arrays into chunks.

# Save ngram list to file for later use
with open('./saved_models/{}_grams.txt'.format(model_name), 'wb') as fp:
    pickle.dump(sent_arr_gram, fp)

print("N-gram samples: ")
for line in islice(sent_arr_gram, 0, 3):
    print(line)

w2v_model = Word2Vec(min_count=1, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=2)

w2v_model.build_vocab(sent_arr_gram, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
t = time()

w2v_model.train(sent_arr_gram, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.wv.save_word2vec_format('./saved_models/{}_model.bin'.format(model_name), binary=True)
w2v_model.save('./saved_models/{}_model.model'.format(model_name))
w2v_model.init_sims(replace=True)

print(w2v_model.wv.most_similar(positive=["duff"]))
