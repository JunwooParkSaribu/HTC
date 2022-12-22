import numpy as np


def load_data(data_path):
    train_data = np.loadtxt(data_path + "mnist_train.csv",
                            delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv",
                           delimiter=",")
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    train_labels = np.asfarray(train_data[:, :1]).astype(int)
    test_labels = np.asfarray(test_data[:, :1]).astype(int)
    return train_imgs, test_imgs, train_labels, test_labels
