#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   audiofeature+2dcnn.py
@Time    :   2020/09/25 11:00:33
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

data = []

for filename in ALLOFFILES:
    y, sr = librosa.load(filename, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
    features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5))
    data.append(features)

X = np.array(data)
X_2d = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))

def conv_2d(input_dim)
	'''
	Input shape should be (num_of_samples, 40, 5, 1)
	'''
	model = Sequential()
	model.add(Conv2D(64, kernel_size=5, strides=1, padding="same", activation="relu", input_shape=(*input_dim, 1)))
	model.add(MaxPooling2D(padding="same"))
	model.add(Conv2D(128, kernel_size=5,strides=1, padding="same", activation="relu"))
	model.add(MaxPooling2D(padding="same"))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(256, activation="relu"))
	model.add(Dropout(0.3))
	model.add(Dense(512, activation="relu"))
	model.add(Dropout(0.3))
	model.add(Dense(6, activation="softmax"))

	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	model.summary()
	return model