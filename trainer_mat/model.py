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
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.datasets import cifar10
from keras.models import Model
#import cv2
from keras.utils import Sequence
from keras.utils import np_utils
from scipy.io import loadmat
import random
from keras.layers.normalization import BatchNormalization
import subprocess
"""Implements the Keras Sequential model."""


import keras
import pandas as pd
from keras import backend as K
from keras import models
from keras.layers import Input

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import numpy as np
import logging


def model_fn(learning_rate, lam, dropout):
    """Create a Keras Sequential model with layers."""
    input = Input(shape=(60,60,3))

    # 1 block
    model = Dropout(0.2, input_shape=(60, 60, 3))(input)
    model = Conv2D(20, (5, 5), strides=(1, 1), padding='valid',
                     kernel_initializer='he_normal', 
                     kernel_regularizer=keras.regularizers.l2(lam)
                     )(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Dropout(dropout)(model)

    # 2 block
    model = Conv2D(40, (7, 7), strides=(1, 1),
                     padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(lam))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Dropout(dropout)(model)

    # 3 block
    model = Conv2D(80, (11, 11), strides=(1, 1),
                     padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(lam))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(dropout)(model)

    model = Flatten()(model)
    model = Dropout(dropout)(model)
    model = Dense(80,
                  kernel_regularizer=keras.regularizers.l2(lam))(model)
    model = Activation('relu')(model)
    outputs = []
    for i in range(80):
        out = Dropout(dropout, name='out_dropout%s' % i)(model)
        out = Dense(2, kernel_regularizer=keras.regularizers.l2(lam), name='dense_out%s' % i)(out)
        out = Activation('softmax', name='out%s' % i)(out)
        outputs.append(out)
    model = Model(inputs=input,outputs=outputs)
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model

CONST_LIST = [float(_i) for _i in range(80)]

def age_mae(y_true, y_pred):
    y_true = tf.cast(K.argmax(y_true, axis=1), dtype=tf.float32)
    labels = K.constant(CONST_LIST, dtype=tf.float32)
    y_pred = labels * y_pred
    y_pred = K.sum(y_pred, axis=1)
    return K.mean(K.abs(y_true-y_pred), axis=0)


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


def convert_to_ordinal(age):
    age_vecs = []
    for i in range(80):
        if age > i:
            age_vec = np.array([0.0,1.0])
        else:
            age_vec = np.array([1.0,0.0])
        age_vecs.append(age_vec) 
    #print('age=%s, age_vec = %s' % (age, age_vec))
    return age_vecs


class DataSequence(Sequence):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
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
        Y_mini = [Y[i][start_idx:end_idx] for i in range(len(Y))]
    else:
        X_mini = X[start_idx:]
        Y_mini = [Y[i][start_idx:] for i in range(len(Y))]

    return X_mini, Y_mini

def unpack(xy):
    x, y = xy
    return x, y


def load_data(mat_path):
    cmd = 'gsutil cp %s /tmp' % mat_path[0]
    subprocess.check_call(cmd.split())
    filename = mat_path[0].split('/')[-1]
    d = loadmat('/tmp/%s' % filename)

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

def convert_to_column_list(y):
    columns = []
    column_len = len(y[0])
    for j in range(column_len):
        column = []
        for i in range(len(y)):
            one_ele = y[i][j]
            column.append(one_ele)
        column = np.array(column)
        columns.append(column)
    return columns

def create_data(input_file):
    (X_train, y_train), (X_test, y_test) = load_data(input_file)

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
    y_train = [convert_to_ordinal(y_train[i])
                        for i in range(len(y_train))]
    y_train = convert_to_column_list(y_train)
    y_test = [convert_to_ordinal(y_test[i])
                       for i in range(len(y_test))]
    y_test = convert_to_column_list(y_test)
    return X_train, y_train, X_test, y_test, input_shape

def rank_decode(preds):
    ages = []
    for j in range(len(preds[0])):
        age = 0
        for i in range(len(preds)):
            # i番目のrankのフラグ回収
            rank_flag = _rank_decode(preds[i][j])
            age += rank_flag
        ages.append(age)
    return np.array(ages)

def _rank_decode(pred_r):
    if pred_r[1] > 0.5:
        return 1.0
    else:
        return 0.0


if __name__ == '__main__':
    y_train = np.array([15,16])
    y_train = [convert_to_ordinal(y_train[i])
                        for i in range(2)]
    y_train = convert_to_column_list(y_train)
    print('encoded=%s' % y_train)
    y_train = rank_decode(y_train)
    print('decoded=%s' % y_train)

#     x_tr, y_tr, x_t, y_t, input_shape = create_data(['gs://kceproject-1113-ml/ordinal-face/wiki_process_10000.mat'])
#     print(x_tr.shape, input_shape)
#     print(len(y_tr))
#     print(y_tr[0])
#     
#     model = model_fn(learning_rate=0.001, lam=0.0, dropout=0.5)
    #print(model.summary())
    #print(type(np_utils.to_categorical(5, 10)[0]))
#     data = get_meta(
#         ['gs://kceproject-1113-ml/intermediate/csv/path_age.csv-00000-of-00221'])
#    seq = DataSequence(x_tr, y_tr, 64)
#                        input_file=[
#                            u'gs://kceproject-1113-ml/intermediate/csv/path_age.csv-00000-of-00221'],
#                        debug_mode=True,
#                        meta_data=data,
#                        batch_size=32,
#                        data_type='train')
#     x_t, y_t = seq.__getitem__(0)
#     print(x_t.shape)
#     print(len(y_t[0][0]))
#     
#     data=model.evaluate_generator(
#                     seq,
#                     steps=seq.length)
#     print(data)
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
