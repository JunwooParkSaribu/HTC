import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ImagePreprocessor
import Labeling
import ImgGenerator
import DataLoad
import ConvModel
import Callback


data_path = 'data/TrainingSample'
model_path = 'my_model'


if __name__ == '__main__':
    nChannel = 3
    print(f'\nLoading the data...')
    histones = DataLoad.read_files(path=data_path, cutoff=10, chunk=False)
    histones_label = Labeling.make_label(histones, radius=0.45, density=0.4)
    print(f'Image processing...')
    ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=nChannel)
    histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=2)
    zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
    print(f'Number of training items:{len(zoomed_imgs)}, processed shape:{scaled_size}, time scale:{time_scale}\n')

    with ConvModel.tf.device('/cpu:0'):
        print(f'Generator building...')
        gen = ImgGenerator.DataGenerator(zoomed_imgs, histones_label, ratio=0.8)
        print(f'Training set length:{gen.get_size()[0]}, Test set length:{gen.get_size()[1]}')
        del histones_imgs; del histones_label; del histones;
        train_ds = ConvModel.tf.data.Dataset.from_generator(gen.train_generator,
                                                            output_types=(ConvModel.tf.float64, ConvModel.tf.int32),
                                                            output_shapes=((scaled_size[0], scaled_size[1], nChannel), ())
                                                            ).batch(32)
        test_ds = ConvModel.tf.data.Dataset.from_generator(gen.test_generator,
                                                           output_types=(ConvModel.tf.float64, ConvModel.tf.int32),
                                                           output_shapes=((scaled_size[0], scaled_size[1], nChannel), ())
                                                           ).batch(32)
        print(f'Training the data...')
        training_model = ConvModel.HTC()
        training_model.compile(jit_compile=True)
        history = training_model.fit(train_ds, test_ds, epochs=100,
                                     callback=Callback.EarlyStoppingAtMinLoss(patience=50))
        training_model.save(model_path)
    print(history)
