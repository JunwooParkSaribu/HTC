import numpy as np
import ImagePreprocessor


class DataGenerator:
    def __init__(self, histones: dict, amp: int, to_size: tuple, ratio=0.8, shuffle=True):
        self.histones = histones
        self.train_keys = []
        self.test_keys = []
        self.amp = amp
        self.to_size = to_size
        self.keys = list(histones.keys())
        self.size = len(histones)
        self.train_size = int(self.size * ratio)
        self.test_size = self.size - self.train_size
        self.n_class = len(set([histones[key].get_manuel_label() for key in self.keys]))
        n_c_check = [0] * self.n_class
        if shuffle:
            np.random.shuffle(self.keys)

        for i, key in enumerate(self.keys):
            label = self.histones[key].get_manuel_label()
            if n_c_check[label] < int(self.train_size/self.n_class):
                self.train_keys.append(key)
                n_c_check[label] += 1
            else:
                self.test_keys.append(key)

    def get_keys(self):
        return self.keys

    def get_size(self):
        return self.train_size, self.test_size

    def get_scaled_size(self):
        return self.to_size

    def train_generator(self):
        train_i = 0
        while train_i < self.train_size:
            trainable_histone = {}
            histone_id = self.train_keys[train_i]
            trainable_histone[histone_id] = self.histones[histone_id]
            histones_imgs, img_size, time_scale =\
                ImagePreprocessor.preprocessing(trainable_histone, img_scale=10, amp=self.amp)
            zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=self.to_size)
            yield zoomed_imgs[histone_id], self.histones[histone_id].get_manuel_label()
            train_i += 1

    def test_generator(self):
        test_i = 0
        while test_i < self.test_size:
            test_histone = {}
            histone_id = self.test_keys[test_i]
            test_histone[histone_id] = self.histones[histone_id]
            histones_imgs, img_size, time_scale =\
                ImagePreprocessor.preprocessing(test_histone, img_scale=10, amp=self.amp)
            zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=self.to_size)
            yield zoomed_imgs[histone_id], self.histones[histone_id].get_manuel_label()
            test_i += 1


def conversion(histones, training_set, keylist=None, batch_size=1000, eval=True):
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
                train_Y.append(histones[keys[i]].get_manuel_label())
                i += 1
                if i % batch_size == 0:
                    break
            yield train_X.copy(), train_Y.copy()
            train_X = []
            train_Y = []
