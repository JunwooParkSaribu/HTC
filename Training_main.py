import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ImagePreprocessor
import Labeling
import ImgGenerator
import DataLoad
import ConvModel
import Callback


data_path = 'data/TrainingData'
model_path = 'my_model'


if __name__ == '__main__':
    print(f'\nLoading the data...')
    histones = DataLoad.read_files(path=data_path, cutoff=10)
    histones_label = Labeling.make_label(histones, radius=0.35, density=0.4)
    print(f'Image processing...')
    histones_channel, nChannel = ImagePreprocessor.make_channel(histones, immobile_cutoff=0.5, hybrid_cutoff=25)
    histones_imgs, img_size, time_scale = \
        ImagePreprocessor.preprocessing(histones, histones_channel, img_size=10, amplif=2, channel=nChannel)
    zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
    print(f'Number of training items:{len(zoomed_imgs)}, processed shape:{scaled_size}, time scale:{time_scale}\n')

    with ConvModel.tf.device('/cpu:0'):
        print(f'Generator building...')
        gen = ImgGenerator.DataGenerator(zoomed_imgs, histones_label, ratio=0.9)
        print(f'Training set length:{gen.get_size()[0]}, Test set length:{gen.get_size()[1]}')
        del histones_imgs; del histones_label; del histones; del histones_channel
        train_ds = ConvModel.tf.data.Dataset.from_generator(gen.train_generator,
                                                            output_types=(ConvModel.tf.float64, ConvModel.tf.int32),
                                                            output_shapes=((scaled_size, scaled_size, nChannel), ())
                                                            ).batch(32)
        test_ds = ConvModel.tf.data.Dataset.from_generator(gen.test_generator,
                                                           output_types=(ConvModel.tf.float64, ConvModel.tf.int32),
                                                           output_shapes=((scaled_size, scaled_size, nChannel), ())
                                                           ).batch(32)
        print(f'Training the data...')
        training_model = ConvModel.HTC()
        training_model.compile(jit_compile=True)
        history = training_model.fit(train_ds, test_ds, epochs=100,
                                     callback=Callback.EarlyStoppingAtMinLoss(patience=10))
        training_model.save(model_path)
    print(history)
