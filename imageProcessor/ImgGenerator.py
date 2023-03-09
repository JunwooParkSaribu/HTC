from imageProcessor import ImagePreprocessor
import numpy as np


class DataGenerator:
    def __init__(self, histones: dict, amp: int, to_size: tuple, ratio=0.8, split_size=50, shuffle=True):
        self.histones = histones
        self.train_keys = []
        self.test_keys = []
        self.train_split = []
        self.test_split = []
        self.amp = amp
        self.to_size = to_size
        self.keys = list(histones.keys())
        self.size = len(histones)
        self.train_size = int(self.size * ratio)
        self.test_size = self.size - self.train_size
        self.n_class = len(set([histones[key].get_manuel_label() for key in self.keys]))
        self.split_num = int(self.size/split_size) if self.size % split_size == 0 else int(self.size/split_size) + 1
        self.train_split_indices = []
        self.test_split_indices = []
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

        self.train_split_indices = [int(i * len(self.train_keys)/self.split_num) for i in range(1, self.split_num)]
        self.test_split_indices = [int(i * len(self.test_keys) / self.split_num) for i in range(1, self.split_num)]

        self.train_split = np.split(np.array(self.train_keys), self.train_split_indices)
        self.test_split = np.split(np.array(self.test_keys), self.test_split_indices)

    def get_keys(self):
        return self.keys

    def get_size(self):
        return self.train_size, self.test_size

    def get_scaled_size(self):
        return self.to_size

    def train_generator(self):
        for i in range(len(self.train_split)):
            train_histones = {}
            histones_id = self.train_split[i]
            for histone_id in histones_id:
                train_histones[histone_id] = self.histones[histone_id]
            histones_imgs, img_size, time_scale =\
                ImagePreprocessor.preprocessing(train_histones, img_scale=10, amp=self.amp)
            zoomed_imgs, _ = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=self.to_size)
            for histone_id in histones_id:
                yield zoomed_imgs[histone_id], self.histones[histone_id].get_manuel_label()

    def test_generator(self):
        for i in range(len(self.test_split)):
            test_histones = {}
            histones_id = self.test_split[i]
            for histone_id in histones_id:
                test_histones[histone_id] = self.histones[histone_id]
            histones_imgs, img_size, time_scale =\
                ImagePreprocessor.preprocessing(test_histones, img_scale=10, amp=self.amp)
            zoomed_imgs, _ = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=self.to_size)
            for histone_id in histones_id:
                yield zoomed_imgs[histone_id], self.histones[histone_id].get_manuel_label()


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
