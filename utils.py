import numpy as np
import itertools
import hashlib

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

def chunks(iterable, size):
    it = iter(iterable)
    chunk = list(itertools.islice(it, size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, size))

def plot_length_vs_accuracy(bins, data_tuples, pred, truth, max_seq_len, title):
    correct_bins = [0] * bins
    total_bins = [0] * bins
    bin_size = (max_seq_len // bins)

    for seq, pred, truth in zip([d[0] for d in data_tuples if d[2] == 'val'], pred, truth):
        bin = int(((len(seq) - 1) / max_seq_len) * bins)
        total_bins[bin] += 1
        if pred == truth:
            correct_bins[bin] += 1
    accuracy = [correct / total for correct, total in zip(correct_bins, total_bins)]
    plt.bar(np.arange(bins), accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Sequence Size')

    plt.title(title)
    plt.xticks(np.arange(bins), ["%d-%d" % (bin_size * i, bin_size * (i + 1)) for i in range(bins)])

def get_split(string, test_split = 0.1, validation_split = 0.1):
    string_hash = hashlib.md5(string.encode('utf-8')).digest()
    prob = int.from_bytes(string_hash[:2], byteorder='big') / 2**16
    if prob < test_split:
        return 'test'
    elif prob > 1 - validation_split:
        return 'val'
    else:
        return 'train'
