import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Conv2D, AveragePooling2D, Input
from tensorflow.keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping


print("TensorFlow version:", tf.__version__)


class LCI(Model):
    def __init__(self):
        super(LCI, self).__init__()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.conv1 = Conv2D(filters=8, kernel_size=(10, 10), strides=(10, 10), activation='relu')
        self.conv2 = Conv2D(filters=8, kernel_size=(5, 5), activation='relu')
        self.conv3 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(3, activation='softmax')
        self.batch1 = BatchNormalization()
        self.activ = Activation("softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.batch1(x)
        x = self.activ(x)
        return x

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self(images, training=True)
            loss = self.loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def fit(self, train_ds, test_ds, EPOCHS):
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.loss_object

            for images, labels in train_ds:
                self.train_step(images, labels)

            for test_images, test_labels in test_ds:
                self.test_step(test_images, test_labels)

            print(
                f'Epoch {epoch + 1 : >4} | '
                f'Loss:{self.train_loss.result() : <8.5f}, '
                f'Accuracy:{self.train_accuracy.result() * 100 : <8.5f}, '
                f'Test Loss:{self.test_loss.result() : <8.5f}, '
                f'Test Accuracy:{self.test_accuracy.result() * 100 : <8.5f}'
            )


"""
def define_model(input_shape):
    # input
    model_input = Input(shape=input_shape)

    # first convolution layer
    model_output = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='Cov1')(model_input)
    model_output = Flatten()(model_output)

    model_output = Dense(10, activation=softmax)(model_output)
    model_output = BatchNormalization()(model_output)
    model_output = Activation("softmax")(model_output)

    return model_input, model_output

def fit_model(model_input, model_output, x_train, y_train, model_path):
    # specify input and output
    model = Model(inputs=model_input,
                  outputs=model_output)

    model.summary()

    # define loss function and optimizer
    model.compile(loss=SparseCategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=0.01),
                  metrics=METRICS)

    # save the best performing model
    checkpointer = ModelCheckpoint(filepath=model_path,
                                   monitor='val_loss',
                                   verbose=0,
                                   save_best_only=True,
                                   mode='min')

    # model training
    model.fit(x_train, y_train,
              batch_size=500,
              epochs=100,
              verbose=2,
              callbacks=[checkpointer],
              validation_split=0.2,
              use_multiprocessing=True)
"""