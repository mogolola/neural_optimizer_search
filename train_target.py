import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from my_optimizer import my_optimizer
from keras import backend as K



class Conv():
    def __init__(self, convsess):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        self.X_train = X_train.astype('float32') / 255
        self.X_test = X_test.astype('float32') /255
        self.y_train = np_utils.to_categorical(y_train, num_classes=10)
        self.y_test = np_utils.to_categorical(y_test, num_classes=10)
        K.set_session(convsess)

        self.model = Sequential()

        # Conv layer 1 output shape (32, 32, 32)
        self.model.add(Convolution2D(
            batch_input_shape=(None, 32, 32, 3),
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',  # Padding method
            data_format='channels_first',
        ))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        # Pooling layer 1 (max pooling) output shape (32, 16, 16)
        self.model.add(MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
            data_format='channels_first',
        ))

        # Conv layer 2 output shape (64, 16, 16)
        self.model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_first'))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        # Pooling layer 2 (max pooling) output shape (64, 8, 8)
        self.model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

        # Fully connected layer 1 input shape (64 * 8 * 8) = (3136), output shape (1024)
        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))

        # Fully connected layer 2 to shape (10) for 10 classes
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

    def train_one_epoche(self, lr, action):
        optimizer = my_optimizer(lr=lr, strings=action)
        #optimizer = SGD()
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print('Training ------------')
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=64, )

    def train(self, lr, action, epochs=5):
        optimizer = my_optimizer(lr=lr, strings=action)
        #optimizer = SGD()
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print('Training ------------')
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=64, )

    def test(self):
        print('\nTesting ------------')
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)

        print('\ntest loss: ', loss)
        print('\ntest accuracy: ', accuracy)
        return accuracy




