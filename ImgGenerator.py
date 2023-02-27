import numpy as np


class DataGenerator:
    def __init__(self, inputs, ratio=0.8, shuffle=True):
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []
        self.keys = list(inputs.keys())
        self.size = len(inputs)
        self.split_index = int(self.size * ratio)
        self.n_class = len(set(inputs[key].get_manuel_label() for key in self.keys))
        n_c_check = [0] * self.n_class

        if shuffle:
            np.random.shuffle(self.keys)

        if inputs[self.keys[0]].get_manuel_label() is not None:
            for i, key in enumerate(self.keys):
                img = inputs[key]
                label = inputs[key].get_manuel_label()
                if n_c_check[label] < int(self.split_index/self.n_class):
                    self.train_X.append(img)
                    self.train_Y.append(label)
                    n_c_check[label] += 1
                else:
                    self.test_X.append(img)
                    self.test_Y.append(label)
        else:
            for i, key in enumerate(self.keys):
                self.test_X.append(inputs[key])

    def get_keys(self):
        return self.keys

    def get_size(self):
        return len(self.train_X), len(self.test_X)

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


def conversion(training_set, keylist=None, batch_size=1000, eval=True):
    size = len(training_set)
    train_X = []
    train_Y = []
    if keylist is None:
        keys = list(training_set.keys())
    else:
        keys = keylist

    if not eval:
        i = 0
        while True:
            if i == size:
                del train_X
                del train_Y
                return
            while i < size:
                train_X.append(training_set[keys[i]])
                i += 1
                if i % batch_size == 0:
                    break
            yield train_X.copy()
            train_X = []
    else:
        i = 0
        while True:
            if i == size:
                del train_X
                del train_Y
                return
            while i < size:
                train_X.append(training_set[keys[i]])
                train_Y.append(training_set[keys[i]].get_manuel_label())
                i += 1
                if i % batch_size == 0:
                    break
            yield train_X.copy(), train_Y.copy()
            train_X = []
            train_Y = []
