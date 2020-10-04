import re
import pandas as pd 
from time import time
from collections import defaultdict
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

import spacy
import logging
import sys 

logging.basicConfig(stream=sys.stdout, format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

t = time()

df = pd.read_csv('./data/simpsons_dataset.csv')

df.isnull().sum()

df = df.dropna().reset_index(drop=True)
print(df.isnull().sum())


nlp = spacy.load('en', disable=['ner', 'parser'])


def cleaning(doc):
    # Begin lemmatization and removing stopwords
    txt = [token.lemma_ for token in doc if not token.is_stop]

    #Remove any sentence less than 2 words
    if len(txt) > 2:
        return ' '.join(txt)
    
# Remove special char's
quick_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])

# https://spacy.io/usage/spacy-101#pipelines
txt = [cleaning(doc) for doc in nlp.pipe(quick_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean: {} min'.format(round((time()- t)/60,2)))

df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
print(df_clean.shape)

#https://radimrehurek.com/gensim/models/phrases.html
# Automatically detect common phrases – aka multi-word expressions, word n-gram collocations – from a stream of sentences.
sent = [row.split() for row in df_clean['clean']]

phrases = Phrases (sent, min_count=30, progress_per=10000)

bigram = Phraser(phrases)

sentences = bigram[sent]

w2v_model = Word2Vec(min_count=20, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=2)

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.wv.save_word2vec_format('./saved_models/model.bin', binary=True)
w2v_model.init_sims(replace=True)

print(w2v_model.wv.most_similar(positive=["homer"]))