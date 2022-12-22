import numpy as np


class DataGenerator:
    def __init__(self, inputs, labels, ratio=0.8, shuffle=True):
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []
        self.keys = list(inputs.keys())
        self.size = len(inputs)
        self.split_index = int(self.size * ratio)
        if shuffle:
            np.random.shuffle(self.keys)
        for i, key in enumerate(self.keys):
            if i < self.split_index:
                self.train_X.append(inputs[key])
                self.train_Y.append(labels[key])
            else:
                self.test_X.append(inputs[key])
                self.test_Y.append(labels[key])

    def train_generator(self):
        i = 0
        while i < len(self.train_X):
            yield self.train_X[i], self.train_Y[i]
            i += 1

    def test_generator(self):
        i = 0
        while i < len(self.test_X):
            yield self.test_X[i], self.test_Y[i]
            i += 1


def split_shuffle(training_set, label_set=None, ratio=1, shuffle=True):
    size = len(training_set)
    split_index = size * ratio
    test_X = []
    train_X = []
    train_Y = []
    test_Y = []
    keys = list(training_set.keys())
    for i in range(size):
        if i < split_index:
            train_X.append(training_set[keys[i]])
            if label_set != None:
                train_Y.append(label_set[keys[i]])
        else:
            test_X.append(training_set[keys[i]])
            if label_set != None:
                test_Y.append(label_set[keys[i]])
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)
