from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer

import ast
import hashlib

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
y = [d[1] for d in data]

vocab_size = len(tokenizer.word_docs)

split = int(0.2*len(data))

x_train, y_train = x[split:], y[split:]
x_test, y_test = x[:split], y[:split]

x_train = sequence.pad_sequences(x_train)
x_test = sequence.pad_sequences(x_test)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

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
