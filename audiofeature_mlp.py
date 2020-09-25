#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   audiofeature_mlp.py
@Time    :   2020/09/25 11:18:57
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

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tqdm.notebook as tq

import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import *

def generate_STFT(file_list, ispca=False, mode='stft', n_fft=512, n_overlap=256, n_mel=128):
    feature = []
    label = []
    
    for file in tq.tqdm(file_list):
        # read audio data
        y, sr = librosa.load(file, sr=None)
        name = file.split('\\')[-2]
        # extract features
        if mode == 'stft':
            spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_overlap, win_length=n_fft)).T
            leng = n_fft//2+1
        if mode == 'mel':
            spec = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=n_overlap, n_mels=n_mel, power=2).T
            leng = n_mel
        if ispca is True:
            pca_sk = PCA(n_components=1)
            flattened = pca_sk.fit_transform(spec.T)
        elif ispca is False:
            flattened = np.mean(spec, axis=0)
        # get feature and label
        feature.append(flattened)
        if name in dict_.keys():
            label.append(dict_[name])
        else: label.append(-1)
    feature = np.array(feature).reshape((-1, leng))
    feature = normalize(feature, norm='max', axis=1)
    label = np.array(label)
    return feature, label

def get_model(inp_dim=257):
    # build model
    model = Sequential()
    model.add(Dense(256, input_shape=(inp_dim, )))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
    # model.summary()
    return model