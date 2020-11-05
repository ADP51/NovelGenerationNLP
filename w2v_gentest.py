
import numpy as np
import gensim
import pickle
import string
from itertools import islice

from unidecode import unidecode
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file
from gensim.models import Word2Vec

model_file = './saved_models/model_model.model'
grams_file = './saved_models/model_grams.txt'

word_model = Word2Vec.load(model_file)
with open(grams_file, "rb") as fp:
    grams = pickle.load(fp)

pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')
for word in ['homer', 'donut', 'duff', 'mayor']:
    most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
    print('  %s -> %s' % (word, most_similar))


def word2idx(word):
    return word_model.wv.vocab[word].index


def idx2word(idx):
    return word_model.wv.index2word[idx]


print('\nPreparing the data for LSTM...')
train_x = np.zeros([len(grams), 80], dtype=np.int32)
train_y = np.zeros([len(grams)], dtype=np.int32)
for i, sentence in enumerate(grams):
    for t, word in enumerate(sentence[:-1]):
        train_x[i, t] = word2idx(word)
    train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print('\nTraining LSTM...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


def sample(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_next(text, num_generated=16):
    word_idxs = [word2idx(word) for word in text.lower().split()]
    for i in range(num_generated):
        prediction = model.predict(x=np.array(word_idxs))
        idx = sample(prediction[-1], temperature=0.7)
        word_idxs.append(idx)
    return ' '.join(idx2word(idx) for idx in word_idxs)


def on_epoch_end(epoch, _):
    print('\nGenerating text after epoch: %d' % epoch)
    texts = [
        'homer',
        'duff',
        'donut',
        'mayor',
        'bart',
        'hello',
        'kill'
    ]
    for text in texts:
        sample = generate_next(text)
        print('%s... -> %s' % (text, sample))


model.fit(train_x, train_y,
          batch_size=128,
          epochs=80,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
