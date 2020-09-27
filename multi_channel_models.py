#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   muti_channel_models.py
@Time    :   2020/09/25 11:02:45
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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn import preprocessing
import scipy
from scipy import stats
from scipy import signal
import pickle
import tables
import tqdm.notebook as tq
from scipy import stats
import statsmodels.api as sm
from scipy.stats import mode
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import noisereduce as nr
from skimage import transform
from sklearn.metrics import accuracy_score
import gammatone.gtgram
from spafe.features.gfcc import gfcc

import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Permute
from keras.optimizers import adam, RMSprop
from keras import regularizers
from keras.models import load_model
from keras.callbacks import TensorBoard


def creat_features(raw_sig, standardize=False):
	'''
	generate 4 channels 128*128 time-frequency feature: chromagram, chroma_cq, mfcc, gfcc
	'''
    chromagram = librosa.feature.chroma_stft(y=raw_sig, sr=sr, n_fft=1024, hop_length=512, n_chroma=128).T
    chroma_cq = librosa.feature.chroma_cqt(y=raw_sig, sr=sr, hop_length=512, n_chroma=128).T
    mfcc_ = librosa.feature.mfcc(y=raw_sig, sr=sr, n_fft=1024, hop_length=512, n_mfcc=128).T
    gfcc_ = gfcc(sig=raw_sig, fs=sr, nfft=1024, win_len=1024/sr, win_hop=508/sr, nfilts=256, num_ceps=128)
    if gfcc_.shape[0] > mfcc_.shape[0]:
        gfcc_ = gfcc_[: mfcc_.shape[0], :]
    elif gfcc_.shape[0] < mfcc_.shape[0]:
        chromagram = chromagram[: gfcc_.shape[0], :]
        chroma_cq = chroma_cq[: gfcc_.shape[0], :]
        mfcc_ = mfcc_[: gfcc_.shape[0], :]
#     print(chromagram.shape, chroma_cq.shape, mfcc_.shape, gfcc_.shape)
#     print(np.max(chromagram), np.min(chromagram), np.max(chroma_cq), np.min(chroma_cq), np.max(mfcc_), np.min(mfcc_), np.max(gfcc_), np.min(gfcc_))
    if standardize is True:
        scaler = MinMaxScaler()
        chromagram = scaler.fit_transform(chromagram)
        chroma_cq = scaler.fit_transform(chroma_cq)
        mfcc_ = scaler.fit_transform(mfcc_)
        gfcc_ = scaler.fit_transform(gfcc_)
#         print(np.max(chromagram), np.min(chromagram), np.max(chroma_cq), np.min(chroma_cq), np.max(mfcc_), np.min(mfcc_), np.max(gfcc_), np.min(gfcc_))
    assert (chromagram.shape == chroma_cq.shape == mfcc_.shape == gfcc_.shape)
    return np.dstack((chromagram, chroma_cq, mfcc_, gfcc_))

def generate_multi_channel(file_name, frame_len=8.18, frame_overlap=7, augmentation=False):
    features = []
    # generate features 2D
    y, sr = librosa.load(file_name, sr=16000)
    
    frame_len = int(frame_len * sr); frame_overlap = int(frame_overlap * sr)
    if augmentation is True:
        y_noise = y + 0.009*np.random.normal(0, 1, len(y))
        y_roll = np.roll(y, int(sr/10))
        y_time_stch = librosa.effects.time_stretch(y, 0.8)
        y_pitch_sf = librosa.effects.pitch_shift(y, sr, n_steps=-5)
        for y_ in [y, y_noise, y_roll, y_time_stch, y_pitch_sf]:
            if len(y_) < frame_len:
                y_pad = np.hstack((y_, y_[-(frame_len - len(y_)): ]))
                fea = creat_features(y_pad)
                features.append(fea)
            else:
                for i in range(0, len(y_)-frame_len, frame_len-frame_overlap):
                    y_tmp = y_[i: i+frame_len]
                    fea = creat_features(y_tmp)
                    features.append(fea)
    else:
        if len(y) < frame_len:
            y_pad = np.hstack((y, y[-(frame_len - len(y)): ]))
            fea = creat_features(y_pad)
            features.append(fea)
        else:
            for i in range(0, len(y)-frame_len, frame_len-frame_overlap):
                y_tmp = y[i: i+frame_len]
                fea = creat_features(y_tmp)
                features.append(fea)
    return features

def generate_spec(file_name, sr=16000, frame_len=4.1, frame_overlap=3, n_fft=1024, hop_length=512, shape=(128, 128)):
	'''
	devide the oribinal STFTs into real and imaginary part, or magnitude and phase part
	'''
    specs = []
    # generate melspectrogram using librosa
    y, sr = librosa.load(file_name, sr=sr)

    spec = librosa.core.stft(y=y, n_fft=n_fft, hop_length=hop_length, window='hanning')
    spec = spec[1: , : ]

    prop = hop_length / sr; frame_len = int(frame_len / prop); frame_overlap = int(frame_overlap / prop)

    for i in range(0, spec.shape[1]-frame_len, frame_len-frame_overlap):
        spec_ = spec[:, i: i+frame_len]
        spec_real, spec_imag = spec_.real, spec_.imag
        magnitude = np.sqrt(spec_real**2 + spec_imag**2)
        phase = np.arctan2(spec_real, spec_imag)
        spec_2 = np.concatenate((magnitude.reshape(*shape, 1), phase.reshape(*shape, 1)), axis=2)
        specs.append(spec_2)

    return specs

def mfc_dcnn(input_dim):
	'''
	multi-channel CNN model, you can modify input_dim based on your input shape
	'''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1, 3), padding='same', input_shape=(*input_dim, 4)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(32, kernel_size=(5, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(5, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(64, kernel_size=(1, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(1, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(64, kernel_size=(5, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(5, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(128, kernel_size=(5, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=(5, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Conv2D(256, kernel_size=(5, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(256, kernel_size=(5, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(256, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(6, kernel_regularizer=regularizers.l2(0.01), activation='softmax'))
    model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
    return model
