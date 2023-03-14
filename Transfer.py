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
import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

do_fine_tuning = True
report_path = './result/pred_wholecells_by_cutoff/cutoff5_model7_lab.csv'
model_dir = './model/model7_lab'
BATCH_SIZE = 32


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
# Labeling.make_label(histones, radius=0.4, density=0.6)
histones = Labeling.label_from_report(histones, report_path)
# histones = DataSimulation.make_simulation_data(number=6)
# DataSave.save_simulated_data(histones, './data/SimulationData/27000_simulated_data.trxyt')
# histones = DataLoad.file_distrib(paths=[f'{cur_path}/data/SimulationData/old/30_simulated_data.trxyt'], cutoff=2, chunk=False)[0]
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
training_model = load_model(model_dir, compile=False)

model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=(500, 500) + (3,)),
    hub.KerasLayer(training_model, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(3,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model.build((500, 500)+(3,))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=['accuracy'],
    run_eagerly=True)

steps_per_epoch = gen.get_size()[0] // BATCH_SIZE
validation_steps = gen.get_size()[1] // BATCH_SIZE
hist = model.fit(
    train_ds,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps).history

saved_model_path = f"./model/transfer_model_{0}"
tf.saved_model.save(model, saved_model_path)
