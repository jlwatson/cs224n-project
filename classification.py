from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer

import ast
import hashlib
import numpy as np

TEST_SPLIT = 0.2
BATCH_SIZE = 32

EMBEDDING_DIM = 128

print('Loading data...')
data = []
for line in open('data/shakespeare/sample.data').readlines()[1:]:
    data.append(ast.literal_eval(line))


tokenizer = Tokenizer()
texts = [d[0] for d in data]
tokenizer.fit_on_texts(texts)
x = tokenizer.texts_to_sequences(texts)

y_vals = [d[1] for d in data]
y = np.zeros((len(y_vals), len(set(y_vals))))
for i, v in enumerate(y_vals):
    y[i, v] = 1.0

vocab_size = len(tokenizer.word_docs) + 1 # somehow getting an extra val in there

split = int(0.2*len(data))

x_train, y_train = x[split:], y[split:]
x_test, y_test = x[:split], y[:split]

x_train = sequence.pad_sequences(x_train)
x_test = sequence.pad_sequences(x_test)

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM))
model.add(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=15,
          validation_split=0.2)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=BATCH_SIZE)
print('Test score:', score)
print('Test accuracy:', acc)
