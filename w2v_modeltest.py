import re
import pandas as pd
import pickle
from time import time
from itertools import islice
import string
import textwrap
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
    docs = unidecode(file.read())

# TODO: Make this number larger (may have to optimize) and implement some system to prevent words being cut off.
#  As it is, many words are being split or truncated

sentence_len = 80

# Cleans and divides sentences into sentence arrays (arrays of arrays of words)
# sent_arr_raw = [[word for word in doc.lower().translate(str.maketrans('', '', string.punctuation))
#                                    .split()[:max_sentence_len]] for doc in docs]

print("Raw text: ")
for line in islice(docs, 0, 3):
    print(line)

# TODO: Investigate possible benefits of keeping punctuation, to get better spacy output
# Further cleaning, lower-casing and removal of punctuation
# (except for periods, exclamation and question points, and apostrophes)
# sent_arr_raw = [doc.lower().translate(str.maketrans('', '', string.punctuation)).rstrip() for doc in docs]
sent_arr_raw = docs.lower().translate(str.maketrans('', '', '~@#$%^&*()+=_",/\\:;{}[]<>')).rstrip()
sent_arr_raw = sent_arr_raw.replace('\n', ' ').replace('-', ' ')

print("Cleaned samples: ")
print(sent_arr_raw[:80], '...')
print(sent_arr_raw[80:160], '...')
print(sent_arr_raw[160:240], '...')

sentence_half = int(sentence_len / 2)
sent_arr_wrap = textwrap.wrap(sent_arr_raw, sentence_len) + textwrap.wrap(sent_arr_raw[sentence_half:], sentence_len)

print("Wrapped samples: ")
for line in islice(sent_arr_wrap, 0, 3):
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


def cleaning_ext(doc):
    # Begin lemmatization and removing stopwords
    txt = [token.lemma_ for token in nlp(doc)]

    # Remove any sentence less than 2 words
    if len(txt) > 2:
        return ' '.join(txt)


# https://spacy.io/usage/spacy-101#pipelines
sent_arr_lem = [cleaning_ext(doc) for doc in sent_arr_wrap]

print('Time to clean: {} min'.format(round((time()- t)/60,2)))
t = time()

print("Lemmatized samples: ")
for line in islice(sent_arr_lem, 0, 3):
    print(line)

sent_arr_split = [row.split() for row in sent_arr_lem if row]

phrases = Phrases(sent_arr_split, min_count=30, progress_per=10000)

bigram = Phraser(phrases)

# Trim ngram sentence length and remove empty lists
sent_arr_gram = [sentence[:sentence_len] for sentence in bigram[sent_arr_split] if sentence]

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

w2v_model.train(sent_arr_gram, total_examples=w2v_model.corpus_count, epochs=60, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.wv.save_word2vec_format('./saved_models/{}_model.bin'.format(model_name), binary=True)
w2v_model.save('./saved_models/{}_model.model'.format(model_name))
w2v_model.init_sims(replace=True)

print('similar to \'duff\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["duff"])])

print('similar to \'homer\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["homer"])])

print('similar to \'donut\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["donut"])])

print('similar to \'santa\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["santa"])])

print('similar to \'nuclear\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["nuclear"])])
