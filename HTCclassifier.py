import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import sys
import numpy as np
import DataLoad
import DataSave
import ImagePreprocessor
import ImgGenerator
import ReadParam
import ProgressBar
from keras.models import load_model
from tensorflow import device


data_path = 'data/1_WT-H2BHalo_noIR/whole cells/Concatenation_20220217_20220301_20220601_H2B_Halo_before_entire_Cell'
model_path = 'my_model'
img_save_path = 'result/image'
report_save_path = 'result'


def predict(gen, scaled_size, nChannel, progress_i, progress_total):
    y_predict = []
    y_predict_proba = []

    for batch_num in range(99999):
        batch = next(gen, -1)
        if batch == -1:
            break
        test_X = np.array(batch).reshape((len(batch), scaled_size[0], scaled_size[1], nChannel))
        with device('/cpu:0'):
            result = HTC_model.predict(test_X, verbose=0)
        y_predict.extend([np.argmax(x) for x in result])
        y_predict_proba.extend([np.max(x) for x in result])
        progress_i += 1
        ProgressBar.printProgressBar(progress_i, progress_total)

        del batch
    return y_predict, y_predict_proba, progress_i


def making_image(histones, zoomed_imgs, scaled_size, amp, img_num, shuffle):
    histone_ids = []
    for histone in histones:
        if histones[histone].get_predicted_proba() < 0.5:
            histone_ids.append(histone)
            img_num -= 1
        if img_num == 0:
            break
    if shuffle:
        np.random.shuffle(histone_ids)

    for histone in histone_ids:
        trajectory = histones[histone].get_trajectory()
        histone_first_pos = [int(trajectory[0][0] * (10 ** amp)),
                             int(trajectory[0][1] * (10 ** amp))]
        ImagePreprocessor.img_save(zoomed_imgs[histone], histones[histone], scaled_size,
                                   histone_first_pos=histone_first_pos,
                                   amp=amp, path=img_save_path)


def main_pipe(full_histones, amp, nChannel, batch_size, make_image=False, img_num=10, shuffle=True):
    total_n_histone = 0
    for g in full_histones:
        total_n_histone += len(list(g.keys()))
    print(f'Total number of histones after cutting off : {total_n_histone}')
    print(f'Predicting...')
    progress_i = 0
    progress_total = int(total_n_histone / params['batch_size']) + 1
    ProgressBar.printProgressBar(progress_i, progress_total)

    for g_num, histones in enumerate(full_histones):
        ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=nChannel)
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=amp)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
        histone_key_list = list(zoomed_imgs.keys())

        gen = ImgGenerator.conversion(zoomed_imgs, keylist=histone_key_list, batch_size=batch_size, eval=False)
        batch_y_predict, batch_y_predict_proba, progress_i = predict(gen, scaled_size, nChannel, progress_i, progress_total)

        for index, histone in enumerate(histone_key_list):
            histones[histone].set_predicted_label(batch_y_predict[index])
            histones[histone].set_predicted_proba(batch_y_predict_proba[index])

        if make_image:
            making_image(histones, zoomed_imgs, scaled_size, amp,
                         img_num=int(img_num/len(full_histones)), shuffle=shuffle)


if __name__ == '__main__':
    make_image = True

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
    main_pipe(full_histones, params['amp'], params['nChannel'], params['batch_size'], make_image)

    print(f'Making reports... ', end=' ')
    DataSave.save_report(full_histones, path=report_save_path)
    print(f'Done.')
