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

#from sklearn.metrics import confusion_matrix
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt


model_path = 'my_model'
data_path = 'data/1_WT-H2BHalo_noIR/whole cells/20220217_h2b halo_before_irradiation_entire_Cell'


if __name__ == '__main__':
    print('python script working dir : ', os.getcwd())
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
        model_path = cur_path + '/' + model_path
        data_path = cur_path + '/' + data_path
    print(model_path)
    print(data_path)

    immobile_cutoff = 0.118
    print(f'Loading data...')
    histones = read_data.read_files(path=data_path)
    histones_label = make_label.make_label(histones, immobile_cutoff, path=data_path)
    print(f'Image processing...')
    histones_imgs, img_size = img_preprocess.preprocessing(histones, img_size=8, amplif=2, channel=1)

    print(f'Reshaping data...')
    test_X, test_Y, a, b = split_shuffle.split_shuffle(histones_imgs, histones_label)
    test_X = test_X.reshape((test_X.shape[0], img_size, img_size, 1))


    """
    print("Before loading data =", datetime.now().strftime("%H:%M:%S"))
    train_X, test_X, train_Y, test_Y = load_data.load_data(data_path)
    print("After loading data =", datetime.now().strftime("%H:%M:%S"))
    img_size = 28  # width and length
    test_X = test_X.reshape((test_X.shape[0], img_size, img_size, 1))
    """

    final_model = load_model(model_path)
    final_model.summary()

    with tf.device('/cpu:0'):
        y_predict = np.array([np.argmax(x) for x in final_model.predict(test_X)])

    print('Accuracy = ', np.sum([1 if x == 0 else 0 for x in (test_Y.reshape(-1) - y_predict)])/float(y_predict.shape[0]))

    """
    cm = confusion_matrix(test_Y, y_predict)
    cm_df = pd.DataFrame(cm,
                         index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                         columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.figure()
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    """
