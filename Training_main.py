import os
import sys
import time
import git
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
from imageProcessor import ImagePreprocessor, ImgGenerator
from fileIO import DataLoad, ReadParam
from model import ConvModel, Callback
import matplotlib.pyplot as plt
from physics import TrajectoryPhy
from label import Labeling
from keras.models import load_model


data_path = './data/TrainingSample'
model_path = './model'
report_path = './result/pred_wholecells_by_cutoff/cutoff5_model7_lab.csv'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
    else:
        cur_path = '.'

    gpus = ConvModel.tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                ConvModel.tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = ConvModel.tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    epochs = 200
    params = ReadParam.read(cur_path)
    print(f'\nLoading the data...')
    histones = DataLoad.file_distrib(paths=params['data'], cutoff=params['cut_off'], chunk=False)[0]
    #Labeling.make_label(histones, radius=0.4, density=0.6)
    histones = Labeling.label_from_report(histones, report_path)
    #histones = DataSimulation.make_simulation_data(number=6)
    #DataSave.save_simulated_data(histones, './data/SimulationData/27000_simulated_data.trxyt')
    #histones = DataLoad.file_distrib(paths=[f'{cur_path}/data/SimulationData/6000_simulated_data.trxyt'], cutoff=2, chunk=False)[0]
    histones = TrajectoryPhy.trjaectory_rotation(histones, 4)

    print(f'Channel processing...')
    ImagePreprocessor.make_channel(histones, immobile_cutoff=5, hybrid_cutoff=12, nChannel=params['nChannel'])

    print(f'Generator building...')
    gen = ImgGenerator.DataGenerator(histones, amp=params['amp'], to_size=(500, 500), ratio=0.8, split_size=32)
    print(f'Number of training items:{sum(gen.get_size())}, processed shape:{gen.get_scaled_size()}\n'
          f'Training set length:{gen.get_size()[0]}, Test set length:{gen.get_size()[1]}')
    train_ds = ConvModel.tf.data.Dataset.from_generator(gen.train_generator,
                                                        output_types=(ConvModel.tf.float64, ConvModel.tf.int32),
                                                        output_shapes=((gen.get_scaled_size()[0],
                                                                        gen.get_scaled_size()[1],
                                                                        params['nChannel']), ())
                                                        ).batch(32)
    test_ds = ConvModel.tf.data.Dataset.from_generator(gen.test_generator,
                                                       output_types=(ConvModel.tf.float64, ConvModel.tf.int32),
                                                       output_shapes=((gen.get_scaled_size()[0],
                                                                       gen.get_scaled_size()[1],
                                                                       params['nChannel']), ())
                                                       ).batch(32)
    print(f'Training the data...')
    #training_model = ConvModel.HTC()
    training_model = ConvModel.HTC(load_model(params['model_dir'], compile=False))

    training_model.compile()
    training_model.summary()
    train_history, test_history = training_model.fit(train_ds, test_ds, epochs=epochs,
                                                     callback=Callback.EarlyStoppingAtMinLoss(patience=10),
                                                     trace='test_loss')

    model_name = ReadParam.write_model_info(training_model, model_path, train_history, test_history, len(histones),
                                            f'{time.gmtime().tm_mday}/{time.gmtime().tm_mon}/{time.gmtime().tm_year}, '
                                            f'{time.gmtime().tm_hour + 1}:{time.gmtime().tm_min}')
    print(f'{model_name} saved...')

    # loss history figure save
    plt.figure()
    plt.plot(range(0, len(train_history)), train_history, label='Train loss')
    plt.plot(range(0, len(test_history)), test_history, label='Validation loss')
    plt.legend()
    plt.savefig(f'{model_path}/{model_name}/loss_history.png')

    # automated git push
    try:
        repo = git.Repo(os.getcwd())
        repo.git.add(f'{model_path}/{model_name}')
        repo.index.commit(f'auto - uploaded')
        origin = repo.remote(name='origin')
        existing_branch = repo.heads['main']
        existing_branch.checkout()
        origin.push()
    except Exception as e:
        print('Git upload failed')
        print(e)
