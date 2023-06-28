import numpy as np
from imageProcessor import ImagePreprocessor


class DataGenerator:
    def __init__(self, histones: dict, amp: int, to_size: tuple, ratio=0.8, split_size=32,
                 shuffle=True, train_keys=None, test_keys=None):
        self.histones = histones
        self.train_split = []
        self.test_split = []
        self.amp = amp
        self.to_size = to_size
        self.keys = list(histones.keys())
        self.size = len(histones)
        self.n_class = max(set([histones[key].get_manuel_label() for key in self.keys])) + 1
        self.train_split_indices = []
        self.test_split_indices = []
        n_c_check = [0] * self.n_class

        if train_keys is None and test_keys is None:
            self.train_keys = []
            self.test_keys = []
            for i, key in enumerate(self.keys):
                label = self.histones[key].get_manuel_label()
                if n_c_check[label] < int(int(self.size * ratio)/self.n_class):
                    self.train_keys.append(key)
                    n_c_check[label] += 1
                else:
                    self.test_keys.append(key)
        else:
            self.train_keys = train_keys
            self.test_keys = test_keys

        if shuffle:
            np.random.shuffle(self.train_keys)
            np.random.shuffle(self.test_keys)

        self.train_size = len(self.train_keys)
        self.test_size = len(self.test_keys)

        self.train_split_indices = [int(i * split_size) for i in
                                    range(1, int(self.train_size / split_size) + 1)]
        self.test_split_indices = [int(i * split_size) for i in
                                   range(1, int(self.test_size / split_size) + 1)]

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
                train_histones[histone_id] = self.histones[histone_id].copy()
            histones_imgs, img_size, time_scale =\
                ImagePreprocessor.preprocessing(train_histones, img_scale=10, amp=self.amp)
            zoomed_imgs, _ = ImagePreprocessor.zoom(histones_imgs, to_size=self.to_size)
            for histone_id in histones_id:
                yield zoomed_imgs[histone_id], train_histones[histone_id].get_manuel_label()

    def test_generator(self):
        for i in range(len(self.test_split)):
            test_histones = {}
            histones_id = self.test_split[i]
            for histone_id in histones_id:
                test_histones[histone_id] = self.histones[histone_id].copy()
            histones_imgs, img_size, time_scale =\
                ImagePreprocessor.preprocessing(test_histones, img_scale=10, amp=self.amp)
            zoomed_imgs, _ = ImagePreprocessor.zoom(histones_imgs, to_size=self.to_size)
            for histone_id in histones_id:
                yield zoomed_imgs[histone_id], test_histones[histone_id].get_manuel_label()


def conversion(histones, key_list=None, scaled_size=(500, 500), batch_size=16, amp=2, eval=True):
    train_X = []
    train_Y = []

    if key_list is None:
        keys = list(histones.keys())
    else:
        keys = key_list
    size = len(keys)

    if not eval:
        i = 0
        while True:
            if i == size:
                del train_X
                del train_Y
                return
            while i < size:
                histones_imgs, img_size, _ = \
                    ImagePreprocessor.preprocessing({keys[i]: histones[keys[i]].copy()}, img_scale=10, amp=amp)
                zoomed_imgs, _ = ImagePreprocessor.zoom(histones_imgs, to_size=scaled_size)
                train_X.append(zoomed_imgs[keys[i]])
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
                histones_imgs, img_size, time_scale = \
                    ImagePreprocessor.preprocessing({keys[i]: histones[keys[i]].copy()}, img_scale=10, amp=amp)
                zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, to_size=scaled_size)
                train_X.append(zoomed_imgs[keys[i]])
                train_Y.append(histones[keys[i]].get_manuel_label())
                i += 1
                if i % batch_size == 0:
                    break
            yield train_X.copy(), train_Y.copy()
            train_X = []
            train_Y = []
