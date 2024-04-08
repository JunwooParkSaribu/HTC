import sys
from label import Labeling
from fileIO import DataLoad
from imageProcessor import ImagePreprocessor, ImgGenerator
from keras.models import load_model
from model import ConvModel
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '.'
    data_path = f'./data/TrainingSample/all_data'
    model_path = f'./model/model42'
    report_path = [f'./data/TrainingSample/manuel_label_model38.csv']
    params = DataLoad.read_params(config_path)

    print(f'Main processing...')

    histones = DataLoad.read_files([data_path], cutoff=2, chunk=False)[0]
    histones = Labeling.label_from_reports(histones, report_path, label_header='label')  # 1040

    HTC_model = load_model(model_path, compile=False)
    HTC_model.compile()

    batch_size = params['batch_size']
    nChannel = params['nChannel']
    amp = params['amp']
    hybrid_cutoff = params['hybrid_cutoff']
    immobile_cutoff = params['immobile_cutoff']

    ImagePreprocessor.make_channel(histones, immobile_cutoff=params['immobile_cutoff'],
                                   hybrid_cutoff=params['hybrid_cutoff'], nChannel=params['nChannel'])

    print(f'Generator building...')
    gen = ImgGenerator.DataGenerator(histones, amp=params['amp'], to_size=(500, 500), ratio=0.8, batch_size=batch_size)
    y_predict = []
    y_predict_proba = []

    test_ds = ConvModel.tf.data.Dataset.from_generator(gen.test_generator,
                                                       output_signature=(
                                                           ConvModel.tf.TensorSpec(
                                                               shape=(gen.get_scaled_size()[0],
                                                                      gen.get_scaled_size()[1],
                                                                      params['nChannel']),
                                                               dtype=ConvModel.tf.float64),
                                                           ConvModel.tf.TensorSpec(
                                                               shape=(),
                                                               dtype=ConvModel.tf.int32))
                                                       ).batch(batch_size)
    result = HTC_model.predict(test_ds)
    y_predict.extend([np.argmax(x) for x in result])
    y_predict_proba.extend([np.max(x) for x in result])

    confusion_mat = ConvModel.tf.math.confusion_matrix(gen.test_labels, y_predict, num_classes=3)
    plt.figure()
    sns.heatmap(confusion_mat / np.sum(confusion_mat), annot=True,
                fmt='.2%', cmap='Blues')
    plt.show()
