import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import DataLoad
import Labeling
import ImagePreprocessor
import ImgGenerator
from keras.models import load_model
import tensorflow as tf


model_path = 'my_model'
data_path = 'data/TestSample'


if __name__ == '__main__':
    amplif = 2
    print('python script working dir : ', os.getcwd())
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
        model_path = cur_path + '/' + model_path
        data_path = cur_path + '/' + data_path
    else:
        cur_path = '.'
    print(model_path)
    print(data_path)

    print(f'Loading the data...')
    histones = DataLoad.read_files(path=data_path, cutoff=10)
    histones_label = Labeling.make_label(histones, radius=0.35, density=0.4)

    print(f'Image processing...')
    histones_channel, nChannel = ImagePreprocessor.make_channel(histones, immobile_cutoff=0.5, hybrid_cutoff=25)
    histones_imgs, img_size, time_scale = \
        ImagePreprocessor.preprocessing(histones, histones_channel, img_size=10, amplif=amplif, channel=nChannel)
    zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))

    print(f'Reshaping the data...')
    test_X, test_Y, histone_key_list = ImgGenerator.split(zoomed_imgs, histones_label)
    test_X = test_X.reshape((test_X.shape[0], scaled_size, scaled_size, nChannel))

    HTC_model = load_model(model_path)
    HTC_model.summary()

    print(f'\nInput shape:{test_X.shape}\n')
    with tf.device('/cpu:0'):
        y_predict = np.array([np.argmax(x) for x in HTC_model.predict(test_X)])
    print('Accuracy = ', np.sum([1 if x == 0 else 0 for x in (test_Y.reshape(-1) - y_predict)])/float(y_predict.shape[0]))

    for i, histone in enumerate(histone_key_list):
        histone_first_pos = [int(histones[histone][0][0] * (10 ** amplif)),
                             int(histones[histone][0][1] * (10 ** amplif))]
        channels = histones_channel[histone]
        if histones_label[histone] != y_predict[i]:
            print(f'Name={histone}')
            ImagePreprocessor.img_save(zoomed_imgs[histone], histone, scaled_size,
                                       label=histones_label[histone], pred=y_predict[i],
                                       histone_first_pos=histone_first_pos, amplif=amplif, path='img/pred_imgs/')
