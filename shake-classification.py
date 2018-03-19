import argparse
import ast
from collections import defaultdict
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import sklearn.metrics
import utils

from mkdir_p import mkdir_p

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import imdb
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, GRU
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model

from attention_layer import Attention

BATCH_SIZE = 32
SPLIT_FRACTION = 0.1
TRAIN_EPOCHS = 15

MIN_SEQUENCE_LEN = 4
MAX_SEQUENCE_LEN = 20

TOKENIZER_FILE = "shake_results/tokenizer.pickle"
WEIGHTS_FILE = "shake_results/shake-weights.hdf5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Data file of (line, author, work) tuples. Use generate_data.py in data/shakespeare to generate new datasets.')
    parser.add_argument('--train', help='Run training.', action='store_true')
    parser.add_argument('--evaluate_test', help='Run evaluations on the test split.', action='store_true')
    parser.add_argument('--evaluate_val', help='Run evaluations on the val split.', action='store_true')
    args = parser.parse_args()

    mkdir_p("shake_results")

    # TODO: just load the pickle if it exists
    print("======= Loading Plays =======")
    print()

    with open(args.data, 'r') as data_handle:
        all_lines = [l.strip() for l in data_handle.readlines()]

    # strip "// Metadata", extract json metadata object, strip "\n // (<fields>)"
    metadata = json.loads(all_lines[1])
    data = [ast.literal_eval(l) for l in all_lines[4:]]
    texts = [d[0] for d in data]

    authors = metadata["authors"]
    works = metadata["works"]

    # author_id -> author_name
    author_id_map = {a[1]: a[0] for a in authors}
    # work_id -> (work_name, author_id)
    works_id_map = {w["id"]: (w["title"], w["author"]) for w in works}

    lines_by_author_and_work = {}
    for a in author_id_map.keys():
        lines_by_author_and_work[a] = defaultdict(list)

    for d in data:
        lines_by_author_and_work[d[1]][d[2]].append(d[0])

    if os.path.isfile(TOKENIZER_FILE):
        with open(TOKENIZER_FILE, 'rb') as handle:
            texts, lines_by_author_and_work, tokenizer, author_id_map, works_id_map = pickle.load(handle)
    else:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        with open(TOKENIZER_FILE, 'wb') as handle:
            pickle.dump((texts, lines_by_author_and_work, tokenizer, author_id_map, works_id_map), handle, protocol=pickle.HIGHEST_PROTOCOL)

    for author, works in lines_by_author_and_work.items():
        print(author_id_map[author], "has", len(works.keys()), "works...")
        for work, lines in works.items():
            print("    ", str(works_id_map[work][0]) + ":", len(lines), "lines")
        print()

    print(len(tokenizer.word_counts), "words in vocab.")
    print()

    print("======= Generating Data Sequences =======")
    print()

    data_tuples = []
    for a in author_id_map.keys():
        line_lists = [w for w in lines_by_author_and_work[a].values()]
        author_lines = []
        for line_list in line_lists:
            author_lines += line_list
        seqs = tokenizer.texts_to_sequences(author_lines)
        for j, seq in enumerate(seqs):
            for chunk in utils.chunks(seq, MAX_SEQUENCE_LEN):
                if len(chunk) >= MIN_SEQUENCE_LEN:
                    # 0 == shakespeare, plz don't be mad
                    data_tuples.append((chunk, 0 if a == 0 else 1))

    print(len(data_tuples), "data tuples.")
    counts = [0] * len(author_id_map)
    for _, label in data_tuples:
        counts[label] += 1

    print("Shakespeare has", counts[0], "labelled examples.")
    print("Other authors have", counts[1], "labelled examples.")

    random.seed(259812)
    random.shuffle(data_tuples)

    vocab_size = len(tokenizer.word_docs) + 1

    X = sequence.pad_sequences([d[0] for d in data_tuples], maxlen=MAX_SEQUENCE_LEN)
    y = [d[1] for d in data_tuples]

    split = int(SPLIT_FRACTION * len(data_tuples))
    X_train, y_train = X[split*2:], y[split*2:]
    X_val, y_val = X[split:split*2], y[split:split*2]
    X_test, y_test = X[:split], y[:split]

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, 128, mask_zero=False))
    model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
    model.add(Attention(direction="bidirectional"))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    plot_model(model, to_file='shake_results/shake-model.png')
    model.summary()

    if os.path.isfile(WEIGHTS_FILE):
        model.load_weights(WEIGHTS_FILE)

    if args.train:
        print("======= Training Network =======")
        print()
        checkpointer = ModelCheckpoint(monitor='val_acc', filepath=WEIGHTS_FILE, verbose=1, save_best_only=True)
        earlystopping = EarlyStopping(monitor='val_acc', patience=10)
        model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=TRAIN_EPOCHS,
                  validation_data=(X_val, y_val),
                  callbacks=[checkpointer, earlystopping])

    if args.evaluate_val:
        score, acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)
        print("Validation score:", score)
        print("Validation accuracy:", acc)

        with open("shake_results/val-metrics.txt", "w") as f:
            f.write("Val score: %s\nVal accuracy: %s" % (score, acc))

        pred = np.around(model.predict(X_val, batch_size=BATCH_SIZE))
        truth = np.around(y_val)
        cnf_matrix = sklearn.metrics.confusion_matrix(truth, pred)
        utils.plot_confusion_matrix(cnf_matrix, classes=["William Shakespeare", "Period playwrights"], normalize=True, title="Val Split Confusion Matrix")
        plt.savefig('shake_results/shake-val-confusion-matrix.png')
        plt.close()

    if args.evaluate_test:
        score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
        print("Test score:", score)
        print("Test accuracy:", acc)

        pred = np.around(model.predict(X_test, batch_size=BATCH_SIZE))
        truth = np.around(y_test)
        cnf_matrix = sklearn.metrics.confusion_matrix(truth, pred)
        utils.plot_confusion_matrix(cnf_matrix, classes=["William Shakespeare", "Period playwrights"], normalize=True, title="Shakespeare Confusion Matrix")
        plt.savefig('shake_results/shake-test-confusion-matrix.png')
        plt.close()

        # TODO: run with Apocrypha separate?
