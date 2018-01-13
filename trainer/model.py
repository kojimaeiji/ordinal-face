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
import cv2
from keras.utils import Sequence
import random

"""Implements the Keras Sequential model."""

import itertools

import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.utils import np_utils
from keras.backend import relu, sigmoid

from urlparse import urlparse

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

from scipy.io import loadmat
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import pandas as pd

# csv columns in the input file
CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race',
               'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket')

CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''],
                       [''], [0], [0], [0], [''], ['']]

# Categorical columns with vocab size
# native_country and fnlwgt are ignored
CATEGORICAL_COLS = (('education', 16), ('marital_status', 7),
                    ('relationship', 6), ('workclass', 9), ('occupation', 15),
                    ('gender', [' Male', ' Female']), ('race', 5))

CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss',
                   'hours_per_week')

LABELS = [' <=50K', ' >50K']
LABEL_COLUMN = 'income_bracket'

#UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
#    zip(*CATEGORICAL_COLS)[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))


def model_fn(learning_rate=0.1):
    """Create a Keras Sequential model with layers."""
    model = models.Sequential()


    # 1 block
    model.add(Conv2D(20, (5, 5), strides=(1, 1), padding='valid',
                     kernel_initializer='he_normal', batch_input_shape=(None, 60, 60, 3)))
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
    model.add(Conv2D(100, (11, 11), strides=(1, 1),
                     padding='valid', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('sigmoid', name='last'))
    #print(model.summary())

#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])

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


# def to_numeric_features(features):
#   """Convert the pandas input features to numeric values.
#      Args:
#         features: Input features in the data
#           age (continuous)
#           workclass (categorical)
#           fnlwgt (continuous)
#           education (categorical)
#           education_num (continuous)
#           marital_status (categorical)
#           occupation (categorical)
#           relationship (categorical)
#           race (categorical)
#           gender (categorical)
#           capital_gain (continuous)
#           capital_loss (continuous)
#           hours_per_week (continuous)
#           native_country (categorical)
#   """
# 
#   for col in CATEGORICAL_COLS:
#     features = pd.concat([features, pd.get_dummies(features[col[0]], drop_first = True)], axis = 1)
#     features.drop(col[0], axis = 1, inplace = True)
# 
#   # Remove the unused columns from the dataframe
#   for col in UNUSED_COLUMNS:
#     features.pop(col)
# 
#   return features


def get_meta(input_path):
    meta = pd.read_csv(tf.gfile.Open(input_path), header=None)

    return meta


def get_length(input_path):
    full_path,_, _ = get_meta(input_path)
    return len(full_path)

def convert_to_ordinal(age):
    age_vec = np.zeros(shape=(80,), dtype=np.float32)
    for i in range(0, age_vec.shape[0]):
        if age > i:
            age_vec[i] = 1.0
    #print('age=%s, age_vec = %s' % (age, age_vec))
    return age_vec


class DataSequence(Sequence):
    def __init__(self, prefix, input_file, debug_mode, batch_size, data_type='train'):
        # コンストラクタ
        self.data_file_path = input_file
        data = get_meta(input_file)
        if debug_mode:
            train_len = 100
            cv_len = 100
        else:
            train_len = int(len(data)*0.96)
            cv_len = int(len(data)*0.02)

        if data_type == 'train':
            self.data = data[0:train_len]
        elif data_type == 'cv':
            self.data = data[train_len:train_len+cv_len]
        else:
            self.data = data[train_len+cv_len:]
        for i in range(len(self.data)):
            self.data.loc[i, 2] = convert_to_ordinal(self.data.loc[i, 2])     
        self.batch_size = batch_size
        self.length = len(self.data) % batch_size
        self.prefix = prefix

    def __getitem__(self, idx):
        # データの取得実装
        full_path = self.data.loc[:, 0]
        age = self.data.loc[:, 2]
        
        batch_size = self.batch_size
        
        # last batch
        if idx != self.length - 1:
            X = convert_images(self.prefix, full_path[idx*batch_size:(idx+1)*batch_size])
            Y = age[idx*batch_size:(idx+1)*batch_size]
        else:
            X = convert_images(self.prefix, full_path[idx*batch_size:])
            Y = age[idx*batch_size:]
        
        return X, Y

    def __len__(self):
        # バッチ数
        return self.length

    def on_epoch_end(self):
        # epoch終了時の処理
        pass



def convert_images(prefix, full_paths):
    x_train = []
    for full_path in full_paths:
        full_path2 = '%s/%s' % (prefix,full_path)
        full_path2 = full_path2.strip()
        #print('path=%s' % full_path2)
        img = cv2.imread(full_path2, 1)
        if img is None:
            raise Exception('image is None:%s' % (full_path2))
        img = img/255.0
        #print('img shape=%s' % ((img.shape),))
        #img = img1[...,::-1]
        #img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
        x_train.append(img)
    x_train = np.array(x_train)
    #print('x_train shape=%s' % ((x_train.shape),))
    return x_train
        

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
 
    #for full_path, age in input_reader:
    #input_data = input_data.dropna()
    #label = pd.get_dummies(input_data.pop(LABEL_COLUMN))
 
    #input_data = to_numeric_features(input_data)
#     n_rows = full_path.shape[0]
#     return ( (full_path.iloc[[index % n_rows]], age.iloc[[index % n_rows]]) for index in itertools.count() )

if __name__ == '__main__':
    seq = DataSequence(prefix=u'/home/jiman/facedata/imdb/intermediate', input_file=[u'/home/jiman/facedata/imdb/imdb_o.mat'])
    x_t, y_t = seq.__getitem__(0)
    img_mat = x_t[0]
    print('shape=%s' % ((img_mat.shape),))
    model_fn()
#     img_mat = img_mat* 255.0
#     img_mat = img_mat.astype(np.uint8)
#     cv2.imshow('image',img_mat)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    
