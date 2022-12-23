import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import img_preprocess
import training as tr
import make_label
import split_shuffle
import read_data


data_path = 'data/1_WT-H2BHalo_noIR/whole cells/20220217_h2b halo_before_irradiation_entire_Cell'
model_path = 'my_model'


if __name__ == '__main__':
    immobile_cutoff = 0.118
    print(f'Loading the data...')
    histones = read_data.read_files(path=data_path)
    histones_label = make_label.make_label(histones, immobile_cutoff)
    print(f'Image processing...')
    histones_imgs, img_size, time_scale = img_preprocess.preprocessing3D(histones, img_size=5, amplif=2, channel=1)

    with tr.tf.device('/cpu:0'):
        print(f'Generator building...')
        gen = split_shuffle.DataGenerator(histones_imgs, histones_label, ratio=0.9)
        del histones_imgs
        del histones_label
        del histones
        train_ds = tr.tf.data.Dataset.from_generator(gen.train_generator,
                                                     output_types=(tr.tf.float64, tr.tf.int32),
                                                     output_shapes=((img_size, img_size, time_scale, 1), ())).batch(32)
        test_ds = tr.tf.data.Dataset.from_generator(gen.test_generator,
                                                    output_types=(tr.tf.float64, tr.tf.int32),
                                                    output_shapes=((img_size, img_size, time_scale, 1), ())).batch(32)
        print(f'Training the data...')
        training_model = tr.LCI()
        training_model.compile(jit_compile=True)
        history = training_model.fit(train_ds, test_ds, epochs=100)
        training_model.save(model_path)
    print(history)
