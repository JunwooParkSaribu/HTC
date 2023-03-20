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
from physics import DataSimulation, TrajectoryPhy
from sklearn.model_selection import KFold
from model import ConvModel, Callback
import matplotlib.pyplot as plt


report_path = './result/pred_wholecells_by_cutoff/cutoff5_model19.csv'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '.'
    params = ReadParam.read(config_path)

    print(f'Loading the data...', end=' ')
    histones = DataLoad.file_distrib(paths=params['data'], cutoff=params['cut_off'], chunk=False)[0]
    print('Done.')
    histones = Labeling.label_from_report(histones, report_path, equal=False)
    print('Labeling done...')
    histones = TrajectoryPhy.trjaectory_rotation(histones, 8)

    nb_samples = [5000, 10000, 5000]
    new_histones = {}
    label0_keys = []
    label1_keys = []
    for histone in histones:
        if histones[histone].get_manuel_label() == 0 and nb_samples[0] > 0:
            new_histones[histone] = histones[histone].copy()
            nb_samples[0] -= 1
            label0_keys.append(histone)

        if histones[histone].get_manuel_label() == 2 and nb_samples[2] > 0:
            histones[histone].set_manuel_label(0)
            new_histones[histone] = histones[histone].copy()
            nb_samples[2] -= 1
            label0_keys.append(histone)

        if histones[histone].get_manuel_label() == 1 and nb_samples[1] > 0:
            new_histones[histone] = histones[histone].copy()
            nb_samples[1] -= 1
            label1_keys.append(histone)

        if sum(nb_samples) == 0:
            del histones
            break

    epochs = 200
    batch_size = 32
    train_acc = []
    test_acc = []
    ImagePreprocessor.make_channel(new_histones, immobile_cutoff=params['immobile_cutoff'],
                                   hybrid_cutoff=params['hybrid_cutoff'], nChannel=params['nChannel'])
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    np.random.shuffle(label0_keys)
    np.random.shuffle(label1_keys)
    for (train_index_0, test_index_0), (train_index_1, test_index_1) in zip((kf.split(label0_keys)), kf.split((label1_keys))):
        train_keys = []
        test_keys = []
        for index in train_index_0:
            train_keys.append(label0_keys[index])
        for index in train_index_1:
            train_keys.append(label1_keys[index])
        for index in test_index_0:
            test_keys.append(label0_keys[index])
        for index in test_index_1:
            test_keys.append(label1_keys[index])
        train_keys = np.array(train_keys)
        test_keys = np.array(test_keys)

        print(f'Generator building...')
        gen = ImgGenerator.DataGenerator(new_histones, amp=params['amp'], to_size=(500, 500), ratio=0.8,
                                         split_size=batch_size, train_keys=train_keys, test_keys=test_keys)
        print(f'Number of training items:{sum(gen.get_size())}, processed shape:{gen.get_scaled_size()}\n'
              f'Training set length:{gen.get_size()[0]}, Test set length:{gen.get_size()[1]}')
        train_ds = ConvModel.tf.data.Dataset.from_generator(gen.train_generator,
                                                            output_signature=(
                                                                ConvModel.tf.TensorSpec(
                                                                    shape=(gen.get_scaled_size()[0],
                                                                           gen.get_scaled_size()[1],
                                                                           params['nChannel']),
                                                                    dtype=ConvModel.tf.float64),
                                                                ConvModel.tf.TensorSpec(
                                                                    shape=(),
                                                                    dtype=ConvModel.tf.int32))
                                                            ).batch(batch_size, drop_remainder=True)
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
                                                           ).batch(batch_size, drop_remainder=True)
        print(f'Training the data...')
        training_model = ConvModel.HTC(end_neurons=2)
        training_model.build(input_shape=(None, gen.get_scaled_size()[0], gen.get_scaled_size()[1], params['nChannel']))
        training_model.compile(optimizer=ConvModel.tf.keras.optimizers.Adam(learning_rate=1e-5),
                               loss=ConvModel.tf.keras.losses.BinaryCrossentropy())
        history = training_model.fit(train_ds, validation_data=test_ds, epochs=epochs,
                                     callbacks=[Callback.EarlyStoppingAtMinLoss(patience=35)],
                                     trace='test_loss')

        best_epoch = np.argmin(history[1])
        train_acc.append(history[2][best_epoch])
        test_acc.append(history[3][best_epoch])

    # loss history figure save
    plt.figure()
    plt.boxplot([train_acc, test_acc])
    plt.legend()
    plt.savefig(f'./img/box_plot.png')
