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
from corpus import Corpus

import spacy
import logging
import sys

logging.basicConfig(stream=sys.stdout, format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

model_name = 'doyle'
# TODO: Link to Corpus class and allow for import of multiple text files from corpus

t = time()

# with open('./data/simpsons_dataset.txt', encoding='utf-8') as file:
#     docs = unidecode(file.read())

# Open a Corpus object with the corpus directory and load a combined string from an author - Doyle
docs = Corpus('./data/corpus_directory.json', './data/corpus/').author_combined_string('Arthur Conan Doyle')

sentence_len = 40

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
sent_clean = docs.lower().translate(str.maketrans('', '', '~@#$%^&*()+=_",/\\:;{}[]<>')).rstrip()
sent_clean = sent_clean.replace('\n', ' ').replace('-', ' ')

print("Cleaned samples: ")
print(sent_clean[:sentence_len * 2], '...')
print(sent_clean[sentence_len * 2:sentence_len * 4], '...')
print(sent_clean[sentence_len * 4:sentence_len * 6], '...')

# Create secondary array by offsetting original array by half the sentence length
# sentence_half = int(sentence_len / 2)
# sent_arr_wrap = textwrap.wrap(sent_clean, sentence_len) + textwrap.wrap(sent_clean[sentence_half:], sentence_len)

# print("Wrapped samples: ")
# for line in islice(sent_arr_wrap, 0, 3):
#     print(line)

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
    # TODO: Don't lemmatize pronouns - some other system to condense them, while retaining the actual word?
    #  I, you, they? Or do they even need to be condensed?

    # Remove any sentence less than 2 words
    if len(txt) > 2:
        return ' '.join(txt)


# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# https://spacy.io/usage/spacy-101#pipelines
# sent_arr_lem = [cleaning_ext(doc) for doc in sent_arr_wrap]
# Temporarily split the text into very large sections, to prevent overloading spacy
sent_lemma_temp = [cleaning_ext(doc) for doc in textwrap.wrap(sent_clean, 100000)]
sent_lemma = ''.join(sent_lemma_temp)

print('Time to clean: {} min'.format(round((time()- t)/60,2)))
t = time()

print("Lemmatized samples: ")
for line in islice(sent_lemma, 0, 3):
    print(line)

# sent_arr_split = [row.split() for row in sent_arr_lem if row]
sent_split = sent_lemma.split()

# Create double-chunked list of split text to feed into Phraser
sent_split_temp = list(chunks(sent_split, sentence_len)) + list(chunks(sent_split[int(sentence_len/2):], sentence_len))

phrases = Phrases(sent_split_temp, min_count=20, progress_per=10000, threshold=2)

bigram = Phraser(phrases)

# Trim ngram sentence length and remove empty lists
# sent_arr_gram = [sentence[:sentence_len] for sentence in bigram[sent_split] if sentence]
sent_gram = bigram[sent_split]

# Final chunking of sentence list
sent_arr_gram = list(chunks(sent_gram, sentence_len))

# Save ngram list to file for later use
with open('E:/NovelGenerationNLP/test_models/{}_grams.txt'.format(model_name), 'wb') as fp:
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
# w2v_model.wv.save_word2vec_format('./saved_models/{}_model.bin'.format(model_name), binary=True)
w2v_model.save('E:/NovelGenerationNLP/test_models/{}_model.model'.format(model_name))
w2v_model.init_sims(replace=True)

print('similar to \'holmes\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["holmes"])])

print('similar to \'mystery\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["mystery"])])

print('similar to \'gun\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["gun"])])

print('similar to \'deduction\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["deduction"])])

print('similar to \'woman\'')
print([word for word, _ in w2v_model.wv.most_similar(positive=["woman"])])
