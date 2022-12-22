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
    print(f'Loading data...')
    histones = read_data.read_files(path=data_path)
    histones_label = make_label.make_label(histones, immobile_cutoff, path=data_path)
    print(f'Image processing...')
    histones_imgs, img_size = img_preprocess.preprocessing(histones, img_size=8, amplif=2, channel=1)

    #train_X, test_X, train_Y, test_Y = load_data.load_data('data/')
    #img_size = 28  # width and length
    #train_X = train_X.reshape((train_X.shape[0], img_size, img_size, 1))
    #test_X = test_X.reshape((test_X.shape[0], img_size, img_size, 1))

    #train_X, train_Y, test_X, test_Y = split_shuffle.split_shuffle(histones_imgs, histones_label, 0.7, shuffle=True)
    #train_ds = tr.tf.data.Dataset.from_tensor_slices((train_X[:2], train_Y[:2])).batch(32)
    #test_ds = tr.tf.data.Dataset.from_tensor_slices((test_X[:5], test_Y[:5])).batch(32)

    print(f'Generator building...')
    gen = split_shuffle.DataGenerator(histones_imgs, histones_label, ratio=0.9)
    train_ds = tr.tf.data.Dataset.from_generator(gen.train_generator,
                                                 output_types=(tr.tf.float64, tr.tf.int32),
                                                 output_shapes=((img_size, img_size, 1), ())).batch(32)
    test_ds = tr.tf.data.Dataset.from_generator(gen.test_generator,
                                                output_types=(tr.tf.float64, tr.tf.int32),
                                                output_shapes=((img_size, img_size, 1), ())).batch(32)

    print(f'Training data...')
    training_model = tr.LCI()
    training_model.compile(jit_compile=True)
    training_model.fit(train_ds, test_ds,
                       EPOCHS=100
                       )

    training_model.save(model_path)
