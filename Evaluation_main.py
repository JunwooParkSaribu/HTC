import os
import sys
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import DataLoad
import Labeling
import ImagePreprocessor
import ImgGenerator
from keras.models import load_model
import tensorflow as tf


data_path = 'data/TestSample'
model_path = 'my_model'


def predict(gen):
    y_predict = []
    test_Y = []
    for batch_num in range(99999):
        gc.collect()
        batch_X, batch_Y = next(gen, (-1, -1))
        if batch_X == -1 or batch_Y == -1:
            break
        print(f'Predicting batch{batch_num+1}...')
        test_X = np.array(batch_X).reshape((len(batch_X), scaled_size, scaled_size, nChannel))
        with tf.device('/cpu:0'):
            y_predict.extend([np.argmax(x) for x in HTC_model.predict(test_X)])
            test_Y.extend(batch_Y)
    return np.array(test_Y), np.array(y_predict)


if __name__ == '__main__':
    amplif = 2
    batch_size = 1000
    y_predict = []
    test_Y = []
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
    histone_key_list = list(zoomed_imgs.keys())
    del histones_imgs

    print(f'Model loading...')
    HTC_model = load_model(model_path)
    HTC_model.summary()

    print(f'\nConverting the data into generator...')
    gen = ImgGenerator.conversion(zoomed_imgs, histones_label,
                                  keylist=histone_key_list, batch_size=batch_size, eval=True)

    print(f'Number of histones:{len(zoomed_imgs)}, batch size:{batch_size}')
    test_Y, y_predict = predict(gen)

    print('Accuracy = ',
          np.sum([1 if x == 0 else 0 for x in (test_Y.reshape(-1) - y_predict)]) / float(y_predict.shape[0]))

    """
    for i, histone in enumerate(histone_key_list):
        histone_first_pos = [int(histones[histone][0][0] * (10 ** amplif)),
                             int(histones[histone][0][1] * (10 ** amplif))]
        channels = histones_channel[histone]
        if histones_label[histone] != y_predict[i]:
            print(f'Name={histone}')
            ImagePreprocessor.img_save(zoomed_imgs[histone], histone, scaled_size,
                                       label=histones_label[histone], pred=y_predict[i],
                                       histone_first_pos=histone_first_pos, amplif=amplif, path='img/pred_imgs')
    """

