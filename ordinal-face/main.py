'''
Created on Jan 5, 2018

@author: saboten
'''
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D


def main(x, y, batch_size=10, epochs=10):

    model = Sequential()

    # 1 block
    model.add(Conv2D(20, (5, 5), strides=(1, 1), padding='valid',
                     kernel_initializer='he_normal', input_shape=(60, 60, 3)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # 2 block
    model.add(Conv2D(40, (7, 7), strides=(1, 1),
                     padding='valid', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # 3 block
    model.add(Conv2D(80, (11, 11), strides=(1, 1),
                     padding='valid', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(80))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x, y, batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    pass
