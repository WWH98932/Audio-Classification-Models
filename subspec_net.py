#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   subspec_net.py
@Time    :   2020/09/25 11:20:30
@Author  :   Wang Minxuan 
@Version :   1.0.0
@Contact :   mx.wang@cyber-insight.com
@License :   (C)Copyright 2019-2020, CyberInsight
@Desc    :   None
'''

# here put the import lib
import glob
import random
import os
import time
import sys
import string
import io
import cv2

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn import preprocessing
from scipy import signal
import pickle
import random
from tqdm import tqdm
from scipy import stats
# from specAugment import spec_augment_tensorflow

import keras
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Permute
from keras.optimizers import adam
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import regularizers
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def ssnet(input_dim):
    '''
    train multiple local classifier using different part of the spectrogram and then combine them with a global classifier
    for example the shape of each sub-STFTs is (128*4, 128), it will be devided into 4*(128, 128)
    '''
    i = 0
    inputs = []
    outputs = []
    toconcat = []
    Y_test = []
    Y_train = []
    Y_test.append(y_test)
    Y_train.append(y_train)
    inputLayer = Input(shape=(*input_dim, 1))

    while(64*i <= 512 - 128):
        # Create Sub-Spectrogram
        input = Lambda(lambda inputLayer: inputLayer[:, i*64: i*64+128, :, :], output_shape=(128, 128, 1))(inputLayer)
        sub0 = Conv2D(32, kernel_size=(1, 3), padding='same')(input)
        sub0 = LeakyReLU(alpha=0.01)(sub0)
        sub0 = Conv2D(32, kernel_size=(1, 3), padding='same')(sub0)
        sub0 = LeakyReLU(alpha=0.01)(sub0)
        sub0 = Conv2D(32, kernel_size=(1, 1), padding='same')(sub0)
        sub0 = LeakyReLU(alpha=0.01)(sub0)
        sub0 = BatchNormalization()(sub0)
        sub0 = MaxPooling2D(pool_size=(1, 2))(sub0)
        sub1 = Conv2D(32, kernel_size=(3, 1), padding='same')(sub0)
        sub1 = LeakyReLU(alpha=0.01)(sub1)
        sub1 = Conv2D(32, kernel_size=(3, 1), padding='same')(sub1)
        sub1 = LeakyReLU(alpha=0.01)(sub1)
        sub1 = Conv2D(32, kernel_size=(1, 1), padding='same')(sub1)
        sub1 = LeakyReLU(alpha=0.01)(sub1)
        sub1 = BatchNormalization()(sub1)
        sub1 = MaxPooling2D(pool_size=(2, 1))(sub1)
        sub2 = Conv2D(64, kernel_size=(1, 3), padding='same')(sub1)
        sub2 = LeakyReLU(alpha=0.01)(sub2)
        sub2 = Conv2D(64, kernel_size=(1, 3), padding='same')(sub2)
        sub2 = LeakyReLU(alpha=0.01)(sub2)
        sub2 = Conv2D(64, kernel_size=(1, 1), padding='same')(sub2)
        sub2 = LeakyReLU(alpha=0.01)(sub2)
        sub2 = BatchNormalization()(sub2)
        sub2 = MaxPooling2D(pool_size=(1, 2))(sub2)
        sub3 = Conv2D(64, kernel_size=(3, 1), padding='same')(sub2)
        sub3 = LeakyReLU(alpha=0.01)(sub3)
        sub3 = Conv2D(64, kernel_size=(3, 1), padding='same')(sub3)
        sub3 = LeakyReLU(alpha=0.01)(sub3)
        sub3 = Conv2D(64, kernel_size=(1, 1), padding='same')(sub3)
        sub3 = LeakyReLU(alpha=0.01)(sub3)
        sub3 = BatchNormalization()(sub3)
        sub3 = MaxPooling2D(pool_size=(2, 1))(sub3)
        sub4 = Conv2D(128, kernel_size=(5, 5), padding='same')(sub3)
        sub4 = LeakyReLU(alpha=0.01)(sub4)
        sub4 = Conv2D(128, kernel_size=(5, 5), padding='same')(sub4)
        sub4 = LeakyReLU(alpha=0.01)(sub4)
        sub4 = Conv2D(128, kernel_size=(1, 1), padding='same')(sub4)
        sub4 = LeakyReLU(alpha=0.01)(sub4)
        sub4 = BatchNormalization()(sub4)
        sub4 = MaxPooling2D(pool_size=(4, 4))(sub4)
        sub5 = Flatten()(sub4)
        sub5 = Dense(512, kernel_regularizer=regularizers.l2(0.01))(sub5)
        subLayer = LeakyReLU(alpha=0.01)(sub5)
        # Sub-Classifier Layer
        subOutput = Dense(6, activation='softmax')(subLayer)

        outputs.append(subOutput)
        toconcat.append(subLayer)
        Y_test.append(y_test)
        Y_train.append(y_train)
        i += 1

    glob0 = Concatenate()(toconcat)
    glob1 = Dense(256, activation='relu')(glob0)
    glob1 = Dropout(0.3)(glob1)
    glob2 = Dense(128, activation='relu')(glob1)
    glob2 = Dropout(0.3)(glob2)
    glob3 = Dense(64, activation='relu')(glob2)
    glob3 = Dropout(0.3)(glob3)
    # Triplet Loss
#     norm_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(glob3)

    # softmax -- GLOBAL CLASSIFIER
    out = Dense(6, activation='softmax')(glob3)
    outputs.append(out)
    model = Model(inputs=inputLayer, outputs=outputs)
    model.summary()

    model.compile(optimizer=adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, Y_train, Y_test