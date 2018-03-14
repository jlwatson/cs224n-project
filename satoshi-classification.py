from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras import utils

import numpy as np
import matplotlib.pyplot as plt

import glob
import random
import itertools

from utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 32
TRAIN_EPOCHS = 15
MAX_SEQUENCE_LEN = 200
CANDIDATES = ["gavin-andresen", "hal-finney", "jed-mccaleb", "nick-szabo", "roger-ver", "dorian-nakamoto"]

print("======= Loading in Texts =======")
texts = []
texts_by_candidate = {}
for c in CANDIDATES + ['satoshi-nakamoto']:
    texts_by_candidate[c] = []
    for path in glob.iglob("./data/satoshi/%s/*.txt" % c, recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            texts.append(text)
            texts_by_candidate[c].append(text)

for auth, texts in texts_by_candidate.items():
    print (auth, "has", len(texts), "texts...")

print("======= Generating vocabulary =======")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print(len(tokenizer.word_counts), "words in vocab.")

print("======= Generating Data Tuples =======")
def chunks(iterable,size):
    it = iter(iterable)
    chunk = list(itertools.islice(it,size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it,size))

data_tuples = []
for i, c in enumerate(CANDIDATES):
    seqs = tokenizer.texts_to_sequences(texts_by_candidate[c])
    for seq in seqs:
        for chunk in chunks(seq, MAX_SEQUENCE_LEN):
            data_tuples.append((chunk, i))

print (len(data_tuples), 'data tuples.')

random.shuffle(data_tuples)

x = sequence.pad_sequences([d[0] for d in data_tuples], maxlen=MAX_SEQUENCE_LEN)
y = utils.to_categorical([d[1] for d in data_tuples], num_classes=len(CANDIDATES))

vocab_size = len(tokenizer.word_docs)

split = int(0.2*len(data_tuples))
x_train, y_train = x[split:], y[split:]
x_test, y_test = x[:split], y[:split]

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(CANDIDATES), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=TRAIN_EPOCHS,
          validation_split=0.2)

score, acc = model.evaluate(x_test, y_test,
                            batch_size=BATCH_SIZE)
print('Test score:', score)
print('Test accuracy:', acc)

pred = np.argmax(model.predict(x_test, batch_size=BATCH_SIZE), axis=1)
truth = np.argmax(y_test, axis=1)
cnf_matrix = confusion_matrix(truth, pred)
plot_confusion_matrix(cnf_matrix, classes=CANDIDATES, normalize=False,
                      title='Confusion Matrix')
plt.savefig('satoshi-confusion-matrix.png')

print("======= Testing Satoshi Writings =======")
satoshi_seqs = tokenizer.texts_to_sequences(texts_by_candidate['satoshi-nakamoto'])
padded = sequence.pad_sequences(satoshi_seqs)
pred = np.argmax(model.predict(padded, batch_size=BATCH_SIZE), axis=1)
with open("satoshi-results.txt", "w") as f:
    f.write('\n'.join(CANDIDATES[i] for i in pred))
