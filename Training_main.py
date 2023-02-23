import ImagePreprocessor
import Labeling
import ImgGenerator
import DataLoad
import DataSimulation
import ConvModel
import Callback
import ReadParam
import matplotlib.pyplot as plt


data_path = 'data/TrainingSample'
model_path = '24_02_simulated_9000samples'
report_path = 'result/eval_10500samples_training.trxyt.csv'


if __name__ == '__main__':
    epochs = 200
    params = ReadParam.read('.')
    print(f'\nLoading the data...')
    #histones = DataLoad.file_distrib(paths=[data_path], cutoff=params['cut_off'], chunk=False)[0]
    #histones_label = Labeling.make_label(histones, radius=0.45, density=0.4)
    #histones_label = Labeling.label_from_report(histones, report_path)
    histones, histones_label = DataSimulation.make_simulation_data(number=3000)

    print(f'Image processing...')
    ImagePreprocessor.make_channel(histones, immobile_cutoff=0.5, hybrid_cutoff=10, nChannel=params['nChannel'])
    histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=params['amp'])
    zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
    print(f'Number of training items:{len(zoomed_imgs)}, processed shape:{scaled_size}, time scale:{time_scale}\n')

    with ConvModel.tf.device('/cpu:0'):
        print(f'Generator building...')
        gen = ImgGenerator.DataGenerator(zoomed_imgs, histones_label, ratio=0.8)
        print(f'Training set length:{gen.get_size()[0]}, Test set length:{gen.get_size()[1]}')
        del histones_imgs; del histones_label; del histones;
        train_ds = ConvModel.tf.data.Dataset.from_generator(gen.train_generator,
                                                            output_types=(ConvModel.tf.float64, ConvModel.tf.int32),
                                                            output_shapes=((scaled_size[0], scaled_size[1], params['nChannel']), ())
                                                            ).batch(32)
        test_ds = ConvModel.tf.data.Dataset.from_generator(gen.test_generator,
                                                           output_types=(ConvModel.tf.float64, ConvModel.tf.int32),
                                                           output_shapes=((scaled_size[0], scaled_size[1], params['nChannel']), ())
                                                           ).batch(32)
        print(f'Training the data...')
        training_model = ConvModel.HTC()
        training_model.compile(jit_compile=True)
        train_history, test_history = training_model.fit(train_ds, test_ds, epochs=epochs,
                                                         callback=Callback.EarlyStoppingAtMinLoss(patience=50),
                                                         trace='training_test_loss')
        training_model.save(model_path)

    # loss history figure save
    plt.figure()
    plt.plot(range(0, len(train_history)), train_history, label='Train loss')
    plt.plot(range(0, len(test_history)), test_history, label='Validation loss')
    plt.legend()
    plt.savefig('./img/loss_history.png')


