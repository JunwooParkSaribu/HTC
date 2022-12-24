import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import read_data
import make_label
import img_preprocess
from keras.models import load_model
import tensorflow as tf
import split_shuffle


model_path = 'my_model'
data_path = 'data/1_WT-H2BHalo_noIR/whole cells/20220217_h2b halo_before_irradiation_entire_Cell'


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
    histones = read_data.read_files(path=data_path)
    histones_label = make_label.make_label(histones, radius=0.2, density=0.3)

    print(f'Image processing...')
    histones_channel, nChannel = img_preprocess.make_channel(histones, immobile_cutoff=0.5, hybrid_cutoff=25)
    histones_imgs, img_size, time_scale = \
        img_preprocess.preprocessing(histones, histones_channel, img_size=10, amplif=amplif, channel=nChannel)
    zoomed_imgs, scaled_size = img_preprocess.zoom(histones_imgs, size=img_size, to_size=(500, 500))

    print(f'Reshaping the data...')
    test_X, test_Y = split_shuffle.split(zoomed_imgs, histones_label)
    test_X = test_X.reshape((test_X.shape[0], scaled_size, scaled_size, nChannel))

    final_model = load_model(model_path)
    final_model.summary()

    with tf.device('/cpu:0'):
        y_predict = np.array([np.argmax(x) for x in final_model.predict(test_X)])
    print('Accuracy = ', np.sum([1 if x == 0 else 0 for x in (test_Y.reshape(-1) - y_predict)])/float(y_predict.shape[0]))

    """
    for i, histone in enumerate(histones):
        histone_first_pos = [int(histones[histone][0][0] * (10 ** amplif)),
                             int(histones[histone][0][1] * (10 ** amplif))]
        channels = histones_channel[histone]
        if i % 10 == 0:
            print(f'i={i}')
        img_preprocess.img_save(zoomed_imgs[histone], histone, scaled_size,
                                label=histones_label[histone], pred=y_predict[i],
                                histone_first_pos=histone_first_pos, amplif=amplif, path='img/pred_imgs/')
    """

