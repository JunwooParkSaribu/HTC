import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, \
    Activation, Conv2D, AveragePooling2D, Dropout, ReLU, MaxPool2D

print("TensorFlow version:", tf.__version__)


class HTC(keras.Model):
    def __init__(self, end_neurons=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_object = None
        self.optimizer = None
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        #self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        #self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

        self.conv0 = Conv2D(filters=32, kernel_size=(8, 8))
        self.pool0 = MaxPool2D(pool_size=(5, 5))
        self.batch0 = BatchNormalization()
        self.relu_activ0 = ReLU()

        self.conv1 = Conv2D(filters=64, kernel_size=(5, 5))
        self.pool1 = MaxPool2D(pool_size=(3, 3))
        self.batch1 = BatchNormalization()
        self.relu_activ1 = ReLU()

        self.conv2 = Conv2D(filters=128, kernel_size=(2, 2))
        self.pool2 = MaxPool2D(pool_size=(2, 2))
        self.batch2 = BatchNormalization()
        self.relu_activ2 = ReLU()

        self.conv3 = Conv2D(filters=256, kernel_size=(2, 2))
        self.pool3 = MaxPool2D(pool_size=(2, 2))
        self.batch3 = BatchNormalization()
        self.relu_activ3 = ReLU()

        self.conv4 = Conv2D(filters=512, kernel_size=(2, 2))
        self.pool4 = MaxPool2D(pool_size=(2, 2))
        self.batch4 = BatchNormalization()
        self.relu_activ4 = ReLU()
        self.drop = Dropout(0.2)

        self.flatten = Flatten()
        self.d1 = Dense(end_neurons)
        self.activ = Activation('softmax')

    def compile(self, optimizer=None, loss=None, **kwargs):
        super().compile()
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        else:
            self.optimizer = optimizer
        if loss is None:
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            self.loss_object = loss

    @tf.function
    def call(self, inputs, training=None, mask=None):
        print('training?=',training)
        x = self.conv0(inputs)
        print('trainable:',self.conv0.trainable)
        #print(self.conv0.trainable_weights[0])
        x = self.pool0(x)
        x = self.batch0(x)
        x = self.relu_activ0(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch1(x)
        x = self.relu_activ1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch2(x)
        x = self.relu_activ2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batch3(x)
        x = self.relu_activ3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.batch4(x, training=training)
        print('batch4', self.batch4)
        print('batch4', self.batch4.trainable)
        x = self.relu_activ4(x)
        # prevent dropout err NHWC to NCHW
        x = tf.transpose(x, [0, 3, 1, 2])
        x = self.drop(x, training=training)
        print('drop trainable:', self.drop.trainable)
        x = tf.transpose(x, [0, 2, 3, 1])

        x = self.flatten(x)
        x = self.d1(x)
        x = self.activ(x)

        return x

    @tf.function
    def train_step(self, data):
        # Unpack the data
        x, y = data
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            y_pred = self(x, training=True)
            loss = self.loss_object(y_true=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute the metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(y, y_pred)

    @tf.function
    def test_step(self, data):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Tracking the loss
        t_loss = self.loss_object(y, y_pred)
        # Update the metrics
        self.test_loss.update_state(t_loss)
        self.test_accuracy.update_state(y, y_pred)

    def fit(self,
            train_ds=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            trace='test_loss'):

        test_ds = validation_data
        callbacks[0].on_train_begin()

        train_loss_results = []
        test_loss_results = []
        train_accuracy_results = []
        test_accuracy_results = []

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            print('training_start')
            for train_data in train_ds:
                self.train_step(train_data)

            print('test_start')
            for test_data in test_ds:
                self.test_step(test_data)

            train_loss_results.append(self.train_loss.result())
            test_loss_results.append(self.test_loss.result())
            train_accuracy_results.append(self.train_accuracy.result())
            test_accuracy_results.append(self.test_accuracy.result())

            if verbose == 1:
                print(
                    f'Epoch {epoch + 1 : >3} | '
                    f'Loss:{self.train_loss.result() : <9.5f} '
                    f'Accuracy:{self.train_accuracy.result() * 100 : <9.5f} '
                    f'Test Loss:{self.test_loss.result() : <9.5f} '
                    f'Test Accuracy:{self.test_accuracy.result() * 100 : <9.5f} ',
                    end=' '
                )

            # Callback
            if trace == 'training_loss':
                best_weight = callbacks[0].on_epoch_end(
                    epoch=epoch, weights=self.weights, loss=self.train_loss.result())
            elif trace == 'training_test_loss':
                best_weight = callbacks[0].on_epoch_end(
                    epoch=epoch, weights=self.weights, loss=self.train_loss.result() + self.test_loss.result())
            else:
                best_weight = callbacks[0].on_epoch_end(
                    epoch=epoch, weights=self.weights, loss=self.test_loss.result())
            if best_weight is not None:
                self.set_weights(best_weight)
                break

        best_weight = callbacks[0].on_train_end()
        self.set_weights(best_weight)
        return [train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results]
