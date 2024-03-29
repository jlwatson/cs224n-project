from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, GRU
from keras.preprocessing.text import Tokenizer
from keras import utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

from mkdir_p import mkdir_p

import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import random
import itertools
import argparse

import pickle

from utils import plot_confusion_matrix, get_split, plot_length_vs_accuracy
from sklearn.metrics import confusion_matrix
from keras import backend as K

from jinja2 import Environment, FileSystemLoader, select_autoescape
from attention_layer import Attention

from keras.utils import plot_model

BATCH_SIZE = 32
TRAIN_EPOCHS = 15

MIN_SEQUENCE_LEN = 10
MAX_SEQUENCE_LEN = 200

WEIGHTS_FILE = "results/satoshi-weights.hdf5"
CANDIDATES = ["gavin-andresen", "hal-finney", "jed-mccaleb", "nick-szabo", "roger-ver", "craig-steven-wright", "wei-dai"]

TOKENIZER_FILE = "results/tokenizer.pickle"

def is_valid_candidiate(c):
    if c not in CANDIDATES:
        raise argparse.ArgumentTypeError("%s is an invalid candidiate" % c)
    return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Run training.', action='store_true')
    parser.add_argument('--evaluate-test', help='Run evaluations on the test split.', action='store_true')
    parser.add_argument('--evaluate-val', help='Run evaluations on the val split.', action='store_true')
    parser.add_argument('--saliency-map', help='Generate a saliency map for this text.')
    parser.add_argument('--saliency-class', help='Generate a saliency map for this class.', type=is_valid_candidiate)
    parser.add_argument('--activation-map', help='Generate an activation map for this text.')
    args = parser.parse_args()

    mkdir_p("results")

    if os.path.isfile(TOKENIZER_FILE):
        print("======= Loading Tokenizer =======")
        with open(TOKENIZER_FILE, 'rb') as handle:
            texts, texts_by_candidate, tokenizer, reverse_word_map = pickle.load(handle)
    else:
        print("======= Loading in Texts =======")
        texts = []
        texts_by_candidate = {}
        for c in CANDIDATES + ['satoshi-nakamoto']:
            texts_by_candidate[c] = []
            for path in glob.iglob("./data/satoshi/%s/*.txt" % c, recursive=True):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    texts.append(text)
                    texts_by_candidate[c].append((text, path))

        print("======= Generating vocabulary =======")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

        with open(TOKENIZER_FILE, 'wb') as handle:
            pickle.dump((texts, texts_by_candidate, tokenizer, reverse_word_map), handle, protocol=pickle.HIGHEST_PROTOCOL)

    for auth, txts in texts_by_candidate.items():
        print (auth, "has", len(txts), "texts...")

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
        seqs = tokenizer.texts_to_sequences(t for t, p in texts_by_candidate[c])
        for j, seq in enumerate(seqs):
            split = get_split(texts_by_candidate[c][j][1])
            for chunk in chunks(seq, MAX_SEQUENCE_LEN):
                if len(chunk) >= MIN_SEQUENCE_LEN:
                    data_tuples.append((chunk, i, split))

    print (len(data_tuples), 'data tuples.')

    counts = [0] * len(CANDIDATES)
    for _, label, _ in data_tuples:
        counts[label] += 1

    for candidate, count in zip(CANDIDATES, counts):
        print(candidate, "has", count, "labelled examples.")

    random.shuffle(data_tuples)

    def prepare_input_matrix(split):
        return sequence.pad_sequences([d[0] for d in data_tuples if d[2] == split], maxlen=MAX_SEQUENCE_LEN)
    def prepare_output_matrix(split):
        return utils.to_categorical([d[1] for d in data_tuples if d[2] == split], num_classes=len(CANDIDATES))

    vocab_size = len(tokenizer.word_docs)

    x_train, y_train = prepare_input_matrix('train'), prepare_output_matrix('train')
    x_test, y_test = prepare_input_matrix('test'), prepare_output_matrix('test')
    x_val, y_val = prepare_input_matrix('val'), prepare_output_matrix('val')

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('x_val shape:', x_val.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, 128, mask_zero=False))
    model.add(Bidirectional(GRU(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
    model.add(Attention(direction="bidirectional"))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(CANDIDATES), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    plot_model(model, to_file='results/satoshi-model.png')
    model.summary()

    if os.path.isfile(WEIGHTS_FILE):
        model.load_weights(WEIGHTS_FILE)

    if args.train:
        print('======= Training Network =======')
        checkpointer = ModelCheckpoint(monitor='val_acc', filepath=WEIGHTS_FILE, verbose=1, save_best_only=True)
        earlystopping = EarlyStopping(monitor='val_acc', patience=10)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=TRAIN_EPOCHS,
                  validation_data=(x_val, y_val),
                  callbacks=[checkpointer, earlystopping])

    if args.evaluate_val:
        score, acc = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
        print('Val score:', score)
        print('Val accuracy:', acc)

        with open("results/satoshi-val-metrics.txt", "w") as f:
            f.write("Val score: %s\nVal accuracy: %s" % (score, acc))

        pred = np.argmax(model.predict(x_val, batch_size=BATCH_SIZE), axis=1)
        truth = np.argmax(y_val, axis=1)
        cnf_matrix = confusion_matrix(truth, pred)
        plot_confusion_matrix(cnf_matrix, classes=CANDIDATES, normalize=True,
                              title='Satoshi Val Split Confusion Matrix')
        plt.savefig('results/satoshi-val-confusion-matrix.png')
        plt.close()

        plot_length_vs_accuracy(10, data_tuples, pred, truth,
            MAX_SEQUENCE_LEN, "Satoshi Accuracy vs. Sequence Length (Val)")
        plt.close()

    if args.evaluate_test:
        score, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
        print('Test score:', score)
        print('Test accuracy:', acc)

        pred = np.argmax(model.predict(x_test, batch_size=BATCH_SIZE), axis=1)
        truth = np.argmax(y_test, axis=1)
        cnf_matrix = confusion_matrix(truth, pred)
        plot_confusion_matrix(cnf_matrix, classes=CANDIDATES, normalize=True,
                              title="Satoshi Confusion Matrix")
        plt.savefig('results/satoshi-test-confusion-matrix.png')
        plt.close()

        plot_length_vs_accuracy(10, data_tuples, pred, truth,
            MAX_SEQUENCE_LEN, "Satoshi Accuracy vs. Sequence Length")
        plt.close()

        print("======= Testing Satoshi Writings =======")
        candidate_counts = [0] * len(CANDIDATES)
        satoshi_seqs = tokenizer.texts_to_sequences(t for t, p in texts_by_candidate['satoshi-nakamoto'])
        paths = [p for t, p in texts_by_candidate['satoshi-nakamoto']]
        padded = sequence.pad_sequences(satoshi_seqs)
        scores = model.predict(padded, batch_size=BATCH_SIZE)
        print(np.sum(scores, axis=0))

        pred = np.argmax(scores, axis=1)
        with open("results/satoshi-results.txt", "w") as f:
            for i, c in enumerate(pred):
                candidate_counts[c] += 1
                f.write(os.path.basename(paths[i]) + "\t" + CANDIDATES[c] + "\t" + str(scores[i]))
                f.write('\n')

        plt.bar(np.arange(len(CANDIDATES)), candidate_counts)
        plt.ylabel('Documents')
        plt.xlabel('Candidates')
        plt.title('Classification of Satoshi Writings')
        plt.xticks(np.arange(len(CANDIDATES)), CANDIDATES)
        plt.close()

    if args.saliency_map:
        print("======= Generating Saliency Map =======")
        with open(args.saliency_map, "r", encoding="utf-8") as f:
            text = f.read()

        input = model.layers[0].output
        output = model.layers[-1].output[:,CANDIDATES.index(args.saliency_class)]
        grad = K.gradients(output, input)[0]
        saliency = K.sum(K.pow(grad, 2), axis=2)
        compute_fn = K.function([model.layers[0].input, K.learning_phase()], [saliency])

        seq = tokenizer.texts_to_sequences([text])[0]
        data = sequence.pad_sequences([seq])
        saliency_mat = compute_fn([data, 0])[0][0]
        saliency_mat = saliency_mat / np.max(saliency_mat)

        scores = model.predict(data)[0]
        pred = np.argmax(scores)

        env = Environment(
            loader=FileSystemLoader('.'),
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template('saliency-vis-template.html')

        tokens = list(reverse_word_map[id] for id in seq)

        with open("results/saliency-map.html", "wb") as f:
            f.write(template.render(words=zip(tokens, saliency_mat)).encode('utf-8'))

        print("Scores:", scores)
        print("Predicted Class:", CANDIDATES[pred])

    if args.activation_map:
        print("======= Generating Activation Map =======")
        with open(args.activation_map, "r", encoding="utf-8") as f:
            text = f.read()

        activations = model.layers[1].output
        compute_fn = K.function([model.layers[0].input, K.learning_phase()], [activations])

        seq = tokenizer.texts_to_sequences([text])[0]
        data = sequence.pad_sequences([seq])
        activation_mat = compute_fn([data, 0])[0][0]

        env = Environment(loader=FileSystemLoader('.'),
            autoescape=select_autoescape(['html', 'xml']))
        env.globals.update(zip=zip, npmax=np.max, npabs=np.abs)
        template = env.get_template('activation-vis-template.html')
        tokens = list(reverse_word_map[id] for id in seq)

        with open("results/activation-map.html", "wb") as f:
            f.write(template.render(tokens=tokens, activation_mat=activation_mat.T).encode('utf-8'))
