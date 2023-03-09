from imageProcessor import ImagePreprocessor, ImgGenerator
from fileIO import DataLoad, ReadParam
from model import ConvModel, Callback
import matplotlib.pyplot as plt
from physics import TrajectoryPhy


data_path = 'data/TrainingSample'
model_path = 'model/model8'
report_path = 'result/eval_10500samples_training.trxyt.csv'


if __name__ == '__main__':
    epochs = 200
    params = ReadParam.read('.')
    print(f'\nLoading the data...')
    #histones = DataLoad.file_distrib(paths=[data_path], cutoff=params['cut_off'], chunk=False)[0]
    #Labeling.make_label(histones, radius=0.4, density=0.6)
    #Labeling.label_from_report(histones, report_path)
    #histones = DataSimulation.make_simulation_data(number=9000)
    #DataSave.save_simulated_data(histones, './data/SimulationData/27000_simulated_data.trxyt')
    histones = DataLoad.file_distrib(paths=['./data/SimulationData/27000_simulated_data.trxyt'], cutoff=2,
                                     chunk=False)[0]
    histones = TrajectoryPhy.trjaectory_rotation(histones, 4)

    print(f'Channel processing...')
    ImagePreprocessor.make_channel(histones, immobile_cutoff=5, hybrid_cutoff=12, nChannel=params['nChannel'])

    print(f'Generator building...')
    gen = ImgGenerator.DataGenerator(histones, amp=params['amp'], to_size=(10, 10), ratio=0.8)
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
    with ConvModel.tf.device('/device:GPU:0'):
        training_model = ConvModel.HTC()
        #training_model.compile()
        train_history, test_history = training_model.fit(train_ds, test_ds, epochs=epochs,
                                                             callback=Callback.EarlyStoppingAtMinLoss(patience=15),
                                                             trace='training_test_loss')
    training_model.save(model_path)

    # loss history figure save
    plt.figure()
    plt.plot(range(0, len(train_history)), train_history, label='Train loss')
    plt.plot(range(0, len(test_history)), test_history, label='Validation loss')
    plt.legend()
    plt.savefig('./img/loss_history.png')


