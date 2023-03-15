import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import sys
import numpy as np
import ProgressBar
from imageProcessor import MakeImage
from label import Labeling
from fileIO import DataLoad, DataSave, ReadParam
from imageProcessor import ImagePreprocessor, ImgGenerator
from keras.models import load_model
from tensorflow import device


report_path = './result/pred_wholecells_by_cutoff/cutoff5_model14.csv'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '.'
    params = ReadParam.read(config_path)


    print(f'Loading the data...', end=' ')
    histones = DataLoad.file_distrib(paths=params['data'], cutoff=params['cut_off'], chunk=False)[0]
    histones = Labeling.label_from_report(histones, report_path, equal=False)
    print(len(histones))


