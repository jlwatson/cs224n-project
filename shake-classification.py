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
TRAIN_EPOCHS = 5

MIN_SEQUENCE_LEN = 10
MAX_SEQUENCE_LEN = 100

DISPUTED_FILE = 'data/shakespeare/disputed_works_75.data'

RESULT_DIR = "shake_results"

TOKENIZER_FILE = RESULT_DIR + "/tokenizer.pickle"
WEIGHTS_FILE = RESULT_DIR + "/shake-weights.hdf5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Data file of (line, author, work) tuples. Use generate_data.py in data/shakespeare to generate new datasets.')
    parser.add_argument('--train', help='Run training.', action='store_true')
    parser.add_argument('--evaluate_test', help='Run evaluations on the test split.', action='store_true')
    parser.add_argument('--evaluate_val', help='Run evaluations on the val split.', action='store_true')
    parser.add_argument('--evaluate_disputed', help='Run model on disputed W.S. works.', action='store_true')
    args = parser.parse_args()

    mkdir_p(RESULT_DIR)

    if os.path.isfile(TOKENIZER_FILE):
        print("======= Loading Tokenizer =======")
        with open(TOKENIZER_FILE, 'rb') as handle:
            tokenizer, data_tuples, author_id_map, works_id_map, lines_by_author_and_work = pickle.load(handle)

    else:
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

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)

        data_tuples = []
        for author, works in lines_by_author_and_work.items():
            print(author_id_map[author], "has", len(works.keys()), "works...")
            for work, lines in works.items():
                print("    ", str(works_id_map[work][0]) + ":", len(lines), "examples")
                for l in tokenizer.texts_to_sequences(lines):
                    # 0 == shakespeare, plz don't be mad
                    data_tuples.append((l, 0 if author == 0 else 1))
            print()

        with open(TOKENIZER_FILE, 'wb') as handle:
            pickle.dump((tokenizer, data_tuples, author_id_map, works_id_map, lines_by_author_and_work), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(len(tokenizer.word_counts), "words in vocab.")
    print()

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

    plot_model(model, to_file=RESULT_DIR+'/shake-model.png')
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

        with open(RESULT_DIR+"/val-metrics.txt", "w") as f:
            f.write("Val score: %s\nVal accuracy: %s" % (score, acc))

        pred = np.around(model.predict(X_val, batch_size=BATCH_SIZE))
        truth = np.around(y_val)
        cnf_matrix = sklearn.metrics.confusion_matrix(truth, pred)
        utils.plot_confusion_matrix(cnf_matrix, classes=["William Shakespeare", "Period playwrights"], normalize=True, title="Val Split Confusion Matrix")
        plt.savefig(RESULT_DIR+'/shake-val-confusion-matrix.png')
        plt.close()

    if args.evaluate_test:
        score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
        print("Test score:", score)
        print("Test accuracy:", acc)

        pred = np.around(model.predict(X_test, batch_size=BATCH_SIZE))
        truth = np.around(y_test)
        cnf_matrix = sklearn.metrics.confusion_matrix(truth, pred)
        utils.plot_confusion_matrix(cnf_matrix, classes=["William Shakespeare", "Period playwrights"], normalize=True, title="Shakespeare Confusion Matrix")
        plt.savefig(RESULT_DIR+'/shake-test-confusion-matrix.png')
        plt.close()

    if args.evaluate_disputed:
        print()
        print("======= Evaluating Disputed Documents =======")
        print()

        with open(DISPUTED_FILE, 'r') as disputed_data_handle:
            all_lines = [l.strip() for l in disputed_data_handle.readlines()]

        # strip "// Metadata", extract json metadata object, strip "\n // (<fields>)"
        metadata = json.loads(all_lines[1])
        data = [ast.literal_eval(l) for l in all_lines[4:]]
        texts = [d[0] for d in data]

        works = metadata["works"]
        works_id_map = {w["id"]: w["title"] for w in works}

        lines_by_work = defaultdict(list)
        for d in data:
            lines_by_work[d[2]].append(d[0])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)

        for work, lines in lines_by_work.items():
            print(str(works_id_map[work]) + ":", len(lines), "examples")
        print()

        random.seed(259812)


        results = []
        for w in works_id_map.keys():

            data_tuples = []
            seqs = tokenizer.texts_to_sequences(lines_by_work[w])
            for j, seq in enumerate(seqs):
                for chunk in utils.chunks(seq, MAX_SEQUENCE_LEN):
                    if len(chunk) >= MIN_SEQUENCE_LEN:
                        # 0 is NOT a label here, just a placeholder
                        data_tuples.append((chunk, 0))

            random.shuffle(data_tuples)

            # vocab_size = len(tokenizer.word_docs) + 1
            disputed_X = sequence.pad_sequences([d[0] for d in data_tuples], maxlen=MAX_SEQUENCE_LEN)

            pred = np.around(model.predict(disputed_X, batch_size=BATCH_SIZE))
            notws = np.sum(pred)
            ws = pred.shape[0] - notws

            print("Predicting authorship of", works_id_map[w] + "...")
            print("     total passages:", pred.shape[0])
            print("    ", ws, "passages attributed to William Shakespeare")
            print("    ", notws, "passages attributed to other authors")
            print("     classification consensus:", "William Shakespeare" if ws > notws else "Other contemporary authors")
            print()

            results.append((w, ws, notws))

        # make the bar graph
        N = len(results)

        ws_counts = [r[1] for r in results]
        notws_counts = [r[2] for r in results]

        indexes = np.arange(N)
        width = 0.35

        p1 = plt.bar(indexes, ws_counts, width, color=(0.3, 0.45, 1.0))
        p2 = plt.bar(indexes, notws_counts, width, color=(0.3, 1.0, 0.45))
        plt.ylabel('% passages in work')
        plt.xlabel('Works')
        plt.title('Classification of passages in contemporary Shakespearean works with disputed attribution')
        plt.xticks(indexes + width/2.0)
        plt.yticks(np.arange(0, 1.0, 0.05))

        plt.savefig(RESULT_DIR+'/shake-disputed-result-bar.png')
        plt.close()
        

















