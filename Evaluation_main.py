import os
import sys
import numpy as np
import DataLoad
import Labeling
import ImagePreprocessor
import ImgGenerator
import ReadParam
import ProgressBar
from keras.models import load_model
from tensorflow import device


data_path = 'data/1_WT-H2BHalo_noIR/whole cells/Concatenation_20220217_20220301_20220601_H2B_Halo_before_entire_Cell'
model_path = 'my_model'


def predict(gen, scaled_size, nChannel, progress_i, progress_total):
    y_predict = []
    y_predict_proba = []

    for batch_num in range(99999):
        batch_X, batch_Y = next(gen, (-1, -1))
        if batch_X == -1 or batch_Y == -1:
            break
        test_X = np.array(batch_X).reshape((len(batch_X), scaled_size[0], scaled_size[1], nChannel))
        with device('/cpu:0'):
            result = HTC_model.predict(test_X, verbose=0)
        y_predict.extend([np.argmax(x) for x in result])
        y_predict_proba.extend([np.max(x) for x in result])
        progress_i += 1
        ProgressBar.printProgressBar(progress_i, progress_total)

        del batch_X
        del batch_Y
    return y_predict, y_predict_proba, progress_i


def making_image(histones, zoomed_imgs, scaled_size, amp):
    print(f'Generating images...')
    for histone in histones:
        if histones[histone].get_manuel_label() != histones[histone].get_predicted_label():
            trajectory = histones[histone].get_trajectory()
            histone_first_pos = [int(trajectory[0][0] * (10 ** amp)),
                                 int(trajectory[0][1] * (10 ** amp))]
            print(f'Name={histone}')
            ImagePreprocessor.img_save(zoomed_imgs[histone], histones[histone], scaled_size,
                                       histone_first_pos=histone_first_pos, amp=amp, path='img/pred_imgs')


def main_pipe(full_histones, amp, nChannel, batch_size):
    total_n_histone = 0
    for g in full_histones:
        total_n_histone += len(list(g.keys()))
    print(f'Total number of histones after cutting off : {total_n_histone}')
    progress_i = 0
    progress_total = int(total_n_histone / params['batch_size']) + 1
    ProgressBar.printProgressBar(progress_i, progress_total)

    for g_num, histones in enumerate(full_histones):
        # Image Processing
        histones_label = Labeling.make_label(histones, radius=0.45, density=0.4)
        ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=nChannel)
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=amp)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
        histone_key_list = list(zoomed_imgs.keys())

        # Image generator
        gen = ImgGenerator.conversion(zoomed_imgs, histones_label,
                                      keylist=histone_key_list, batch_size=batch_size, eval=True)
        # Prediction
        batch_y_predict, batch_y_predict_proba, progress_i = predict(gen, scaled_size, nChannel, progress_i, progress_total)

        for index, histone in enumerate(histone_key_list):
            histones[histone].set_predicted_label(batch_y_predict[index])
            histones[histone].set_predicted_proba(batch_y_predict_proba[index])
            histones[histone].set_manuel_label(histones_label[histone])

        #making_image(histones, zoomed_imgs, scaled_size, amp)


if __name__ == '__main__':
    print('python script working dir : ', os.getcwd())
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '.'

    params = ReadParam.read(config_path)

    print(f'Loading the data...')
    data = DataLoad.file_distrib(paths=params['data'], cutoff=params['cut_off'],
                                 all=params['all'], group_size=params['group_size'])
    HTC_model = load_model(params['model_dir'])
    HTC_model.summary()

    # Main pipe start.
    for file_num, dt in enumerate(data):
        if params['all']:
            print(f'Predicting all data...')
            main_pipe(dt, params['amp'], params['nChannel'], params['batch_size'])

            # Result analysis
            histones = {}
            for h in dt:
                histones |= h
            miss_classfied = 0
            for i, histone in enumerate(histones):
                if histones[histone].get_predicted_label() != histones[histone].get_manuel_label():
                    miss_classfied += 1
            print(f'Accuracy = {(i - miss_classfied) / i}')

        else:
            print(f'{file_num+1}/{len(data)} Predicting {dt[1]}')
            main_pipe(dt[0], params['amp'], params['nChannel'], params['batch_size'])

            # Result analysis
            histones = {}
            for h in dt[0]:
                histones |= h
            miss_classfied = 0
            for i, histone in enumerate(histones):
                if histones[histone].get_predicted_label() != histones[histone].get_manuel_label():
                    miss_classfied += 1
            print(f'Accuracy = {(i - miss_classfied) / i}')
