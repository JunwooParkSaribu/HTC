import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import DataLoad
import Labeling
import ImagePreprocessor
import ImgGenerator
import ReadParam
import ProgressBar
from keras.models import load_model
import tensorflow as tf


data_path = 'data/TestSample'
model_path = 'my_model'


def predict(gen, scaled_size, nChannel, progress_i, progress_total):
    y_predict = []
    test_Y = []
    for batch_num in range(99999):
        batch_X, batch_Y = next(gen, (-1, -1))
        if batch_X == -1 or batch_Y == -1:
            break
        test_X = np.array(batch_X).reshape((len(batch_X), scaled_size[0], scaled_size[1], nChannel))
        with tf.device('/cpu:0'):
            y_predict.extend([np.argmax(x) for x in HTC_model.predict(test_X, verbose=0)])
            test_Y.extend(batch_Y)
        progress_i += 1
        ProgressBar.printProgressBar(progress_i, progress_total)
        del batch_X
        del batch_Y
    return test_Y, y_predict, progress_i


def making_image(histones, histones_label, y_predict, zoomed_imgs, histone_key_list, scaled_size, amp):
    print(f'Generating images...')
    for i, histone in enumerate(histone_key_list):
        histone_first_pos = [int(histones[histone][0][0] * (10 ** amp)),
                             int(histones[histone][0][1] * (10 ** amp))]
        if histones_label[histone] != y_predict[i]:
            print(f'Name={histone}')
            ImagePreprocessor.img_save(zoomed_imgs[histone], histone, scaled_size,
                                       label=histones_label[histone], pred=y_predict[i],
                                       histone_first_pos=histone_first_pos, amp=amp, path='img/pred_imgs')


def main_pipe(full_histones, amp, nChannel, batch_size):
    total_n_histone = 0
    for g in full_histones:
        total_n_histone += len(list(g.keys()))
    print(f'Total number of histones after cutting off : {total_n_histone}')
    print(f'Predicting...')
    progress_i = 0
    progress_total = int(total_n_histone / params['batch_size']) + 1
    ProgressBar.printProgressBar(progress_i, progress_total)

    y_predict = []
    test_Y = []
    full_histones_key = []

    for g_num, histones in enumerate(full_histones):
        # Image Processing
        histones_label = Labeling.make_label(histones, radius=0.45, density=0.4)
        ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=nChannel)
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=amp)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))

        histone_key_list = list(zoomed_imgs.keys())
        full_histones_key.extend(histone_key_list)

        # Image generator
        gen = ImgGenerator.conversion(zoomed_imgs, histones_label,
                                      keylist=histone_key_list, batch_size=batch_size, eval=True)

        # Prediction
        batch_test_Y, batch_y_predict, progress_i = predict(gen, scaled_size, nChannel, progress_i, progress_total)
        test_Y.extend(batch_test_Y)
        y_predict.extend(batch_y_predict)

        #making_image(histones, histones_label, batch_y_predict, zoomed_imgs, histone_key_list, scaled_size, amp)
    return np.array(test_Y), np.array(y_predict), np.array(full_histones_key)


if __name__ == '__main__':
    print('python script working dir : ', os.getcwd())
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
        model_path = cur_path + '/' + model_path
        data_path = cur_path + '/' + data_path
    else:
        cur_path = '.'
    print(model_path)
    print(data_path)

    params = ReadParam.read(cur_path)

    print(f'Loading the data...')
    full_histones = DataLoad.read_files(path=data_path, cutoff=params['cut_off'], group_size=params['group_size'])  # 16GB RAM
    print(f'If total number of trajectories is bigger than {params["group_size"]},\n'
          f'data will be separated into groups to reduce the memory usage.')

    print(f'Model loading...')
    HTC_model = load_model(model_path)
    HTC_model.summary()

    # Main pipe start.
    test_Y, y_predict, full_histones_key = main_pipe(full_histones, params['amp'], params['nChannel'], params['batch_size'])

    print('Accuracy = ',
          np.sum([1 if x == 0 else 0 for x in (test_Y.reshape(-1) - y_predict)]) / float(y_predict.shape[0]))

