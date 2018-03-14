from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras import utils

import glob
import random
import numpy as np
import itertools

BATCH_SIZE = 32
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
          epochs=15,
          validation_split=0.2)

score, acc = model.evaluate(x_test, y_test,
                            batch_size=BATCH_SIZE)
print('Test score:', score)
print('Test accuracy:', acc)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

pred = np.argmax(model.predict(x_test, batch_size=BATCH_SIZE), axis=1)
truth = np.argmax(y_test, axis=1)
cnf_matrix = confusion_matrix(truth, pred)

plot_confusion_matrix(cnf_matrix, classes=CANDIDATES, normalize=False,
                      title='Normalized confusion matrix')
plt.show()
