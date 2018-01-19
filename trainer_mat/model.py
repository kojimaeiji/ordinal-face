# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from keras.layers.core import Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.datasets import cifar10
#import cv2
from keras.utils import Sequence
from keras.utils import np_utils
from scipy.io import loadmat
import random
"""Implements the Keras Sequential model."""


import keras
import pandas as pd
from keras import backend as K
from keras import models


import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import numpy as np
import logging


def model_fn(learning_rate=0.01):
    """Create a Keras Sequential model with layers."""
    model = models.Sequential()

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
    model.add(Activation('sigmoid', name='last'))

    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


def get_meta(input_path):
    print('read input files')
    files = tf.gfile.Glob(input_path[0])
    print('find input files')

    list_ = []
    for file_ in files:
        print('file=%s' % file_)
        df = pd.read_csv(tf.gfile.Open(file_), index_col=None, header=None)
        print(len(df))
        list_.append(df)

    frame = pd.concat(list_)
    print(len(frame))
    return frame


def get_length(input_path):
    full_path, _, _ = get_meta(input_path)
    return len(full_path)


def convert_to_ordinal(age):
    age_vec = np.zeros(shape=(80,), dtype=np.float64)
    for i in range(0, age_vec.shape[0]):
        if age > i:
            age_vec[i] = 1.0
    #print('age=%s, age_vec = %s' % (age, age_vec))
    return age_vec


class DataSequence(Sequence):
    def __init__(self, x, y, batch_size):
        # コンストラクタ
        #self.data_file_path = input_file
        #data = get_meta(input_file)
        self.x = x
        self.y = y
#         for i in range(len(self.y)):
#             age_vec = convert_to_ordinal(self.y[i])
#             self.y[i] = age_vec
        self.batch_size = batch_size
        self.length = len(self.x) // batch_size if len(
            self.x) % batch_size == 0 else (len(self.x) // batch_size) + 1

    def __getitem__(self, idx):
        # データの取得実装
        logger = logging.getLogger()
        #logger.info('idx=%s' % idx)
        #age = self.data.loc[:, 2]

        batch_size = self.batch_size

        # last batch or not
        if idx != self.length - 1:
            X, Y = convert_to_minibatch(
                self.x, self.y, idx * batch_size, (idx + 1) * batch_size)
        else:
            X, Y = convert_to_minibatch(
                self.x, self.y, idx * batch_size, None)

        return X, Y

    def __len__(self):
        # バッチ数
        return self.length

    def on_epoch_end(self):
        # epoch終了時の処理
        pass


def convert_to_minibatch(X, Y, start_idx, end_idx):
    if end_idx:
        X_mini = X[start_idx:end_idx]
        Y_mini = Y[start_idx:end_idx]
    else:
        X_mini = X[start_idx:]
        Y_mini = Y[start_idx:]

#     _x = []
#     _y = []
#     for x, y in zip(X_mini, Y_mini):
#         _x.append(x)
#         _y.append(y)
#     _x = np.array(_x)
#     _y = np.array(_y)
    return X_mini, Y_mini


def convert_image(prefix, sub_path):
    full_path = '%s/%s' % (prefix, sub_path)
    full_path = full_path.strip()
    #img = cv2.imread(tf.gfile.Open(full_path), 1)
    img_array = np.asarray(bytearray(tf.gfile.FastGFile(
        full_path, 'rb').read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    if img is None:
        return None
    img = img / 255.0
    return img

# def convert_images(prefix, full_paths):
#     x_train = []
#     for full_path in full_paths:
#         full_path2 = '%s/%s' % (prefix,full_path)
#         full_path2 = full_path2.strip()
#         #print('path=%s' % full_path2)
#         img = cv2.imread(full_path2, 1)
#         if img is None:
#             x_train.appene(None)
#             continue
#             #raise Exception('image is None:%s' % (full_path2))
#         img = img/255.0
#         #print('img shape=%s' % ((img.shape),))
#         #img = img1[...,::-1]
#         #img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
#         x_train.append(img)
#     x_train = np.array(x_train)
#     #print('x_train shape=%s' % ((x_train.shape),))
#     return x_train


# def generator_input(input_file, chunk_size):
#     """Generator function to produce features and labels
#          needed by keras fit_generator.
#     """
#     input_reader = pd.read_csv(tf.gfile.Open(input_file[0]),
#     #                            #names=CSV_COLUMNS,
#                                 header=None,
#                                 chunksize=chunk_size,
    #                            na_values=" ?")
    #full_path, age = get_meta(input_file[0])

    # for full_path, age in input_reader:
    #input_data = input_data.dropna()
    #label = pd.get_dummies(input_data.pop(LABEL_COLUMN))

    #input_data = to_numeric_features(input_data)
#     n_rows = full_path.shape[0]
# return ( (full_path.iloc[[index % n_rows]], age.iloc[[index % n_rows]])
# for index in itertools.count() )
def unpack(xy):
    x, y = xy
    return x, y


def load_data():
    mat_path = '/Users/saboten/data/wiki_process_10000.mat'
    d = loadmat(mat_path)

    x, y = d["image"], d["age"][0]
    length = len(y)
    full_mat = [(x[i], y[i]) for i in range(length)]
    random.shuffle(full_mat)
    train_ratio = 0.9
    train_end = int(length * train_ratio)
    full_mat_train = full_mat[0:train_end]
    full_mat_test = full_mat[train_end:]
    test_length = len(full_mat_test)
    print(full_mat_train[0][0].shape)
    x_train = np.array([full_mat_train[i][0] for i in range(train_end)])
    y_train = np.array([full_mat_train[i][1] for i in range(train_end)])
    x_test = np.array([full_mat_test[i][0] for i in range(test_length)])
    y_test = np.array([full_mat_test[i][1] for i in range(test_length)])
    return (x_train, y_train), (x_test, y_test)


def create_data():
    (X_train, y_train), (X_test, y_test) = load_data()

    # データをfloat型にして正規化する
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    img_rows = 60
    img_cols = 60

    # image_data_formatによって畳み込みに使用する入力データのサイズが違う
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(-1, 3, img_rows, img_cols)
        X_test = X_test.reshape(-1, 3, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(-1, img_rows, img_cols, 3)
        X_test = X_test.reshape(-1, img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    #y_train = np_utils.to_categorical(y_train, 10)
#     y_test = np_utils.to_categorical(y_test, 10)
    length_train = len(y_train)
    y_train = np.array([convert_to_ordinal(y_train[i])
                        for i in range(len(y_train))])
    y_test = np.array([convert_to_ordinal(y_test[i])
                       for i in range(len(y_test))])

    return X_train, y_train, X_test, y_test, input_shape


if __name__ == '__main__':
    x_tr, y_tr, x_t, y_t, input_shape = create_data()
    print(x_tr.shape, input_shape)
    print(y_tr.shape)
    print(y_tr[0])
    #print(type(np_utils.to_categorical(5, 10)[0]))
#     data = get_meta(
#         ['gs://kceproject-1113-ml/intermediate/csv/path_age.csv-00000-of-00221'])
#     seq = DataSequence(prefix=u'gs://kceproject-1113-ml/intermediate',
#                        input_file=[
#                            u'gs://kceproject-1113-ml/intermediate/csv/path_age.csv-00000-of-00221'],
#                        debug_mode=True,
#                        meta_data=data,
#                        batch_size=32,
#                        data_type='train')
#     x_t, y_t = seq.__getitem__(0)
#     img_mat = x_t[0]
#     print('shape=%s' % ((img_mat.shape),))
#     print(type(x_t[0][0][0][0]))
#     print(type(y_t[0][0]))
    # model_fn()
#     img_mat = img_mat* 255.0
#     img_mat = img_mat.astype(np.uint8)
#     cv2.imshow('image',img_mat)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
