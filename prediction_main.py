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
    print('python script working dir : ', os.getcwd())
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
        model_path = cur_path + '/' + model_path
        data_path = cur_path + '/' + data_path
    else:
        cur_path = '.'
    print(model_path)
    print(data_path)

    immobile_cutoff = 0.118
    print(f'Loading the data...')
    histones = read_data.read_files(path=data_path)
    histones_label = make_label.make_label(histones, immobile_cutoff)

    print(f'Image processing...')
    histones_imgs, img_size = img_preprocess.preprocessing(histones, img_size=8, amplif=2, channel=1)
    histones_imgs_2D, img_size = img_preprocess.preprocessing(histones, img_size=8, amplif=2, channel=False)

    print(f'Reshaping the data...')
    test_X, test_Y = split_shuffle.split(histones_imgs, histones_label)
    test_X = test_X.reshape((test_X.shape[0], img_size, img_size, 1))

    final_model = load_model(model_path)
    final_model.summary()

    with tf.device('/cpu:0'):
        y_predict = np.array([np.argmax(x) for x in final_model.predict(test_X)])
    print('Accuracy = ', np.sum([1 if x == 0 else 0 for x in (test_Y.reshape(-1) - y_predict)])/float(y_predict.shape[0]))

    histone_names = histones_imgs_2D.keys()
    for i, name in enumerate(histone_names):
        if i == 50:
            break
        sample_img = histones_imgs_2D[name]
        sample_label = histones_label[name]
        sample_pred = y_predict[i]
        zoomed_img, zoom_size = img_preprocess.zoom(sample_img, size=img_size)
        img_preprocess.img_save(zoomed_img, name,
                                img_size=zoom_size, label=sample_label, pred=sample_pred, path=cur_path)

