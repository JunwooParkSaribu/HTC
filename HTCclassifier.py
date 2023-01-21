import os
import sys
import numpy as np
import DataLoad
import DataSave
import ImagePreprocessor
import ImgGenerator
import ReadParam
import ProgressBar
from keras.models import load_model
import tensorflow as tf


data_path = 'data/TestSample'
model_path = 'my_model'
img_save_path = 'result/image'
report_save_path = 'result'


def predict(gen, scaled_size, nChannel, progress_i, progress_total):
    y_predict = []
    for batch_num in range(99999):
        batch = next(gen, -1)
        if batch == -1:
            break
        test_X = np.array(batch).reshape((len(batch), scaled_size[0], scaled_size[1], nChannel))
        with tf.device('/cpu:0'):
            y_predict.extend([np.argmax(x) for x in HTC_model.predict(test_X, verbose=0)])
        progress_i += 1
        ProgressBar.printProgressBar(progress_i, progress_total)
        del batch
    return y_predict, progress_i


def making_image(histones, y_predict, zoomed_imgs, histone_key_list, scaled_size, amp):
    print(f'Generating images...')
    for i, histone in enumerate(histone_key_list):
        trajectory = histones[histone].get_trajectory()
        histone_first_pos = [int(trajectory[0][0] * (10 ** amp)),
                             int(trajectory[0][1] * (10 ** amp))]
        if i % 500 == 0:
            ImagePreprocessor.img_save(zoomed_imgs[histone], histone, scaled_size[0],
                                       pred=y_predict[i], histone_first_pos=histone_first_pos,
                                       amp=amp, path=img_save_path)


def main_pipe(full_histones, amp, nChannel, batch_size, make_image=False):
    total_n_histone = 0
    for g in full_histones:
        total_n_histone += len(list(g.keys()))
    print(f'Total number of histones after cutting off : {total_n_histone}')
    print(f'Predicting...')
    progress_i = 0
    progress_total = int(total_n_histone / params['batch_size']) + 1
    ProgressBar.printProgressBar(progress_i, progress_total)

    y_predict = []
    full_histones_key = []

    for g_num, histones in enumerate(full_histones):
        ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=nChannel)
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=amp)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
        histone_key_list = list(zoomed_imgs.keys())
        full_histones_key.extend(histone_key_list)

        gen = ImgGenerator.conversion(zoomed_imgs, keylist=histone_key_list, batch_size=batch_size, eval=False)
        predicted_y, progress_i = predict(gen, scaled_size, nChannel, progress_i, progress_total)
        y_predict.extend(predicted_y)

        if make_image:
            making_image(histones, predicted_y, zoomed_imgs, histone_key_list, scaled_size, amp)

    return np.array(y_predict), np.array(full_histones_key)


if __name__ == '__main__':
    make_image = False

    print('python script working dir : ', os.getcwd())
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
        model_path = cur_path + '/' + model_path
        data_path = cur_path + '/' + data_path
        img_save_path = cur_path + '/' + img_save_path
        report_save_path = cur_path + '/' + report_save_path
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

    # Main pipe start.
    y_predict, full_histones_key = main_pipe(full_histones, params['amp'], params['nChannel'], params['batch_size'], make_image)

    print(f'Making reports... ', end=' ')
    DataSave.save_report(full_histones, y_predict, full_histones_key, path=report_save_path)
    print(f'Done.')


