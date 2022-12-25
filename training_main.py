import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import img_preprocess
import training as tr
import make_label
import split_shuffle
import read_data


data_path = 'data/TrainingData'
model_path = 'my_model'


if __name__ == '__main__':
    print(f'\nLoading the data...')
    histones = read_data.read_files(path=data_path, cutoff=10)
    histones_label = make_label.make_label(histones, radius=0.35, density=0.5)
    print(f'Image processing...')
    histones_channel, nChannel = img_preprocess.make_channel(histones, immobile_cutoff=0.5, hybrid_cutoff=25)
    histones_imgs, img_size, time_scale = \
        img_preprocess.preprocessing(histones, histones_channel, img_size=10, amplif=2, channel=nChannel)
    zoomed_imgs, scaled_size = img_preprocess.zoom(histones_imgs, size=img_size, to_size=(500, 500))
    print(f'Processed shape:{zoomed_imgs.shape}, Time scale:{time_scale}\n')

    with tr.tf.device('/cpu:0'):
        print(f'Generator building...')
        gen = split_shuffle.DataGenerator(zoomed_imgs, histones_label, ratio=0.9)
        del histones_imgs; del histones_label; del histones; del histones_channel
        train_ds = tr.tf.data.Dataset.from_generator(gen.train_generator,
                                                     output_types=(tr.tf.float64, tr.tf.int32),
                                                     output_shapes=((scaled_size, scaled_size, nChannel), ())).batch(32)
        test_ds = tr.tf.data.Dataset.from_generator(gen.test_generator,
                                                    output_types=(tr.tf.float64, tr.tf.int32),
                                                    output_shapes=((scaled_size, scaled_size, nChannel), ())).batch(32)
        print(f'Training the data...')
        training_model = tr.LCI()
        training_model.compile(jit_compile=True)
        history = training_model.fit(train_ds, test_ds, epochs=100)
        training_model.save(model_path)
    print(history)
