import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, \
    Activation, Conv2D, AveragePooling2D, Dropout, ReLU, MaxPool2D

print("TensorFlow version:", tf.__version__)


class HTC(keras.Model):
    def __init__(self):
        super(HTC, self).__init__()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.conv1 = Conv2D(filters=32, kernel_size=(8, 8))
        self.pool1 = MaxPool2D(pool_size=(5, 5))
        self.batch1 = BatchNormalization()
        self.relu_activ1 = ReLU()

        self.conv2 = Conv2D(filters=64, kernel_size=(5, 5))
        self.pool2 = MaxPool2D(pool_size=(3, 3))
        self.batch2 = BatchNormalization()
        self.relu_activ2 = ReLU()

        self.conv3 = Conv2D(filters=128, kernel_size=(2, 2))
        self.pool3 = MaxPool2D(pool_size=(2, 2))
        self.batch3 = BatchNormalization()
        self.relu_activ3 = ReLU()

        self.conv4 = Conv2D(filters=256, kernel_size=(2, 2))
        self.pool4 = MaxPool2D(pool_size=(2, 2))
        self.batch4 = BatchNormalization()
        self.relu_activ4 = ReLU()

        self.conv5 = Conv2D(filters=512, kernel_size=(2, 2))
        self.pool5 = MaxPool2D(pool_size=(2, 2))
        self.batch5 = BatchNormalization()
        self.relu_activ5 = ReLU()
        self.dropout1 = Dropout(0.2)

        self.flatten = Flatten()
        self.d_fin = Dense(3, activation='softmax')
        self.batch_fin = BatchNormalization()
        self.soft_activ = Activation("softmax")

    def call(self, x):

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
        x = self.batch4(x)
        x = self.relu_activ4(x)

        x = self.conv5(x)
        x = self.pool5(x)
        x = self.batch5(x)
        x = self.relu_activ5(x)
        x = x.transpose(x, [0, 3, 1, 2])
        x = self.dropout1(x)
        x = x.transpose(x, [0, 2, 3, 1])

        x = self.flatten(x)
        x = self.d_fin(x)
        x = self.batch_fin(x)
        x = self.soft_activ(x)

        return x

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self(images, training=True)
            loss = self.loss_object(y_true=labels, y_pred=predictions)

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

    def fit(self, train_ds, test_ds, epochs, callback, trace='test_loss'):
        callback.on_train_begin()

        train_loss_results = []
        test_loss_results = []
        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in train_ds:
                self.train_step(images, labels)

            for test_images, test_labels in test_ds:
                self.test_step(test_images, test_labels)

            train_loss_results.append(self.train_loss.result())
            test_loss_results.append(self.test_loss.result())

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
                best_weight = callback.on_epoch_end(
                    epoch=epoch, weights=self.weights, loss=self.train_loss.result())
            elif trace == 'training_test_loss':
                best_weight = callback.on_epoch_end(
                    epoch=epoch, weights=self.weights, loss=self.train_loss.result() + self.test_loss.result())
            else:
                best_weight = callback.on_epoch_end(
                    epoch=epoch, weights=self.weights, loss=self.test_loss.result())
            if best_weight is not None:
                self.set_weights(best_weight)
                break

        best_weight = callback.on_train_end()
        self.set_weights(best_weight)
        return train_loss_results, test_loss_results
