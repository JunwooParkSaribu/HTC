import sys
import numpy as np
import ProgressBar
from fileIO import DataLoad, DataSave, ReadParam
from label import Labeling
from imageProcessor import ImagePreprocessor, ImgGenerator
from keras.models import load_model
from tensorflow import device


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


def main_pipe(full_histones, scaled_size=(500, 500), immobile_cutoff=5,
              hybrid_cutoff=12, amp=2, nChannel=3, batch_size=32):
    total_n_histone = 0
    for g in full_histones:
        total_n_histone += len(list(g.keys()))
    print(f'Number of histones after cutting off : {total_n_histone}')
    progress_i = 0
    progress_total = int(total_n_histone / params['batch_size']) + 1
    ProgressBar.printProgressBar(progress_i, progress_total)

    for g_num, histones in enumerate(full_histones):
        Labeling.make_label(histones, radius=0.4, density=0.6)
        #Labeling.label_from_report(histones, './result/old_eval_all_35300h2b.csv')

        key_list = list(histones.keys())
        ImagePreprocessor.make_channel(histones, immobile_cutoff=immobile_cutoff,
                                       hybrid_cutoff=hybrid_cutoff, nChannel=nChannel)
        gen = ImgGenerator.conversion(histones, key_list=key_list, scaled_size=scaled_size,
                                      batch_size=batch_size, amp=amp, eval=True)
        batch_y_predict, batch_y_predict_proba, progress_i = predict(gen, scaled_size,
                                                                     nChannel, progress_i, progress_total)
        for index, histone in enumerate(key_list):
            histones[histone].set_predicted_label(batch_y_predict[index])
            histones[histone].set_predicted_proba(batch_y_predict_proba[index])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '.'
    params = ReadParam.read(config_path)

    print(f'Loading the data...')
    full_data = DataLoad.file_distrib(paths=params['data'], cutoff=params['cut_off'], group_size=params['group_size'])

    HTC_model = load_model(params['model_dir'], compile=False)
    HTC_model.compile()
    # Main pipe start.
    print(f'Predicting all data...')
    main_pipe(full_data, scaled_size=(500, 500), immobile_cutoff=params['immobile_cutoff'],
              hybrid_cutoff=params['hybrid_cutoff'], amp=params['amp'],
              nChannel=params['nChannel'], batch_size=params['batch_size'])
    print(f'Making reports... ', end=' ')
    DataSave.save_report(full_data, path=params['save_dir'], all=params['all'], eval=True)
    print(f'Done.')