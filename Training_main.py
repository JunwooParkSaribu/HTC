import sys
import time
from imageProcessor import ImagePreprocessor, ImgGenerator
from fileIO import DataLoad, DataSave
from model import ConvModel, Callback
import matplotlib.pyplot as plt
from physics import TrajectoryPhy
from label import Labeling


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
    else:
        cur_path = '.'

    data_path = f'{cur_path}/data/TrainingSample/all_data'
    model_path = f'{cur_path}/model'
    report_path = [f'{cur_path}/data/TrainingSample/manuel_label_model38.csv']

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

    params = DataLoad.read_params(cur_path)
    epochs = 200
    batch_size = params['batch_size']
    print(f'\nLoading the data...')
    histones = DataLoad.read_files([data_path], cutoff=2, chunk=False)[0]
    histones = Labeling.label_from_reports(histones, report_path, label_header='label')  # 1040
    histones = TrajectoryPhy.trajectory_rotation(histones, 24)

    print(f'Channel processing...')
    ImagePreprocessor.make_channel(histones, immobile_cutoff=params['immobile_cutoff'],
                                   hybrid_cutoff=params['hybrid_cutoff'], nChannel=params['nChannel'])

    print(f'Generator building...')
    gen = ImgGenerator.DataGenerator(histones, amp=params['amp'], to_size=(500, 500), ratio=0.8, split_size=batch_size)
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
    training_model = ConvModel.HTC(end_neurons=3)
    training_model.build(input_shape=(None, gen.get_scaled_size()[0], gen.get_scaled_size()[1], params['nChannel']))
    training_model.summary()
    training_model.compile(optimizer=ConvModel.tf.keras.optimizers.Adam(learning_rate=1e-4))
    history = training_model.fit(train_ds, validation_data=test_ds, epochs=epochs,
                                 callbacks=[Callback.EarlyStoppingAtMinLoss(patience=40),
                                            Callback.LearningRateScheduler()],
                                 trace='test_loss')  # training_loss, training_test_loss, test_loss

    model_name = DataSave.write_model_info(training_model, model_path, history, len(histones),
                                            f'{time.gmtime().tm_mday}/{time.gmtime().tm_mon}/{time.gmtime().tm_year}, '
                                            f'{time.gmtime().tm_hour + 1}:{time.gmtime().tm_min}')
    print(f'{model_name} saved...')

    # loss history figure save
    plt.figure()
    plt.plot(range(0, len(history[0])), history[0], label='Train loss')
    plt.plot(range(0, len(history[1])), history[1], label='Validation loss')
    plt.legend()
    plt.savefig(f'{model_path}/{model_name}/loss_history.png')

    """
    # automated git push
    try:
        repo = git.Repo(os.getcwd())
        origin = repo.remote(name='origin')
        origin.pull()
        repo.git.add(f'{model_path}/{model_name}')
        repo.index.commit(f'auto - uploaded')
        existing_branch = repo.heads['main']
        existing_branch.checkout()
        origin.push()
    except Exception as e:
        print('Git upload failed')
        print(e)
    """