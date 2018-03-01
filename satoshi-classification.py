from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras import utils

import glob
import random

BATCH_SIZE = 32
CANDIDATES = ["gavin-andresen", "hal-finney", "jed-mccaleb", "nick-szabo", "roger-ver"]

texts = []
texts_by_candidate = {}
for c in CANDIDATES:
    texts_by_candidate[c] = []
    for path in glob.iglob("./data/satoshi/%s/*.txt" % c, recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            texts.append(text)
            texts_by_candidate[c].append(text)

print("Generating vocabulary...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print(len(tokenizer.word_counts))

import itertools
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

data_tuples = []
for i, c in enumerate(CANDIDATES):
    seqs = tokenizer.texts_to_sequences(texts_by_candidate[c])
    for seq in seqs:
        # print(seq)
        # print([grouper(seq, MAX_LEN)])
        # os.exit()
        data_tuples.append((seq, i))

random.shuffle(data_tuples)

x = sequence.pad_sequences([d[0] for d in data_tuples], maxlen=200)
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
          epochs=15,
          validation_split=0.2)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=BATCH_SIZE)
print('Test score:', score)
print('Test accuracy:', acc)
