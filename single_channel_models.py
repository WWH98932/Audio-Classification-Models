#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   stft+cldnn.py
@Time    :   2020/09/25 11:09:06
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
from keras_self_attention import SeqSelfAttention
from resnext import ResNext

def generate_mel_spec(file_name, method='stft', frame_len=4.1, frame_overlap=3, 
                      n_mels=128, n_fft=1024, hop_length=512, power=2.0):
    log_mels = []
    # generate melspectrogram using librosa
    y, sr = librosa.load(file_name, sr=16000)
    if method == 'logmel':
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
            )
        # convert melspectrogram to log mel energy
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    elif method == 'stft':
        stft = np.abs(librosa.core.stft(y=y, n_fft=n_fft, hop_length=hop_length, window='hanning'))
        # convert melspectrogram to log mel energy
        log_mel_spectrogram = 20.0 / power * np.log10(stft + sys.float_info.epsilon)
        log_mel_spectrogram = log_mel_spectrogram[1: , : ]
    prop = 512 / 16000; frame_len = int(frame_len / prop); frame_overlap = int(frame_overlap / prop)
    if log_mel_spectrogram.shape[1] < frame_len:
        pad = np.zeros((log_mel_spectrogram.shape[0], frame_len-log_mel_spectrogram.shape[1])) + np.min(log_mel_spectrogram)
        log_mel = np.hstack((log_mel_spectrogram, pad))
        log_mels.append(log_mel)
    else:
        for i in range(0, log_mel_spectrogram.shape[1]-frame_len, frame_len-frame_overlap):
            log_mel = log_mel_spectrogram[:, i: i+frame_len]
            log_mels.append(log_mel)
    return log_mels

def generate_STFT(file_list, isnr=True, trim=True, top_db=20, mode='stft', n_fft=512, n_overlap=256, n_mel=128, shape=128):
    '''
    isnr: noise reduction based on Spectral Gate
    trim: remove the silence part in audio based on relevant db level
    mode: stft or mel spectrogram
    '''
    feature = []
    label = []
    names = []
    
    for file in tq.tqdm(file_list):
        # read audio data
        y, sr = librosa.load(file, sr=16000)
        name = file.split('\\')[-2]
        # noise reduction
        if isnr is True:
            y = nr.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)
        if trim is True:
            #trimming
#             y, index = librosa.effects.trim(y, top_db=10, frame_length=512, hop_length=256)
            non_silent_interval = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=64)
            y_tmp = np.array([])
            for intv in non_silent_interval:
                y_tmp = np.concatenate([y_tmp, y[intv[0]: intv[1]]])
            y = y_tmp
        # extract features
        if mode == 'stft':
            spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_overlap, win_length=n_fft))
            spec = spec[1: ,: ]
#             spec = 10 * np.log10(spec + sys.float_info.epsilon)
        if mode == 'mel':
            spec = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=n_overlap, n_mels=n_mel, power=2)
        # get feature and label
#         spec = transform.resize(spec, (shape, shape))
        if spec.shape[1] <= shape:
            spec_tmp = np.pad(spec, [(0, 0), (0, shape-spec.shape[1])], mode='constant')
            feature.append(spec_tmp)
            if name in dict_.keys():
                label.append(dict_[name])
            else: label.append(-1)
            names.append(file.split('\\')[-1])
        else:
            for i in range(0, spec.shape[1]-shape, shape//2):
                spec_tmp = spec[:, i: i+shape]
                feature.append(spec_tmp)
                if name in dict_.keys():
                    label.append(dict_[name])
                else: label.append(-1)
                names.append(file.split('\\')[-1])
            if spec.shape[1] % shape > 0: 
                feature.append(spec[:, -shape:])
                if name in dict_.keys():
                    label.append(dict_[name])
                else: label.append(-1)
                names.append(file.split('\\')[-1])
    feature = np.array(feature).reshape((-1, shape, shape, 1))
    return feature, np.array(label), names

def loss_plot(loss, val_loss):
    ax = plt.figure(figsize=(30, 10)).add_subplot(1, 1, 1)
    ax.cla()
    ax.plot(loss)
    ax.plot(val_loss)
    ax.set_title("Model loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Train", "Validation"], loc="upper right")
def save_figure( name):
    plt.savefig(name)

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention, self).__init__()
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(attention,self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

def baseline_cldnn(input_dim):
    '''
    input dimension is (128, 128), Frequency domain first
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1, 3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros(), input_shape=(*input_dim, 1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 1), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 1), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Permute((2, 1, 3)))
    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=True)))
    # model.add(attention(return_sequences=True))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=False)))
    # model.add(attention(return_sequences=True))
    # model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(6, kernel_regularizer=regularizers.l2(0.0001), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4, decay=1e-6), metrics=['accuracy'])
    model.summary()
    return model

def baseline_lstm(input_dim):
    '''
    input dimension is (128, 128), Time domain first, need to be transposed
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x[:, :, :, 0], input_shape=(*input_dim, 1)))
    model.add(LSTM(128, dropout=0.1, return_sequences=True))
    model.add(LSTM(128, dropout=0.1, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(128))
    model.add(Dense(6, activation="softmax"))
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
    return model

def ae_extractor(input_dim):
    '''
    convolutional autoencoder based feature extractor, use bottleneck layer as 1d feature
    '''
    # Input
    inp = Input(shape=(*input_dim, 1))
    # Encoder
    e = Conv2D(16, (3, 3), padding='same')(inp)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(32, (3, 3), padding='same')(e)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(64, (3, 3), padding='same')(e)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(128, (3, 3), padding='same')(e)
    e = BatchNormalization()(e)
    e = LeakyReLU(alpha=0.1)(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    l = Flatten()(e)
    l = Dense(e.shape[1] * e.shape[2])(l) # 250
    l = LeakyReLU(alpha=0.1)(l)
    encoded = l
    # Decoder
    d = Reshape((e.shape[1], e.shape[2], 1))(encoded)
    d = Conv2D(128, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(64, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(32, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(16, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(1, (3, 3), padding='same')(d)           
    d = BatchNormalization()(d)
    d = Activation('sigmoid')(d)
    decoded = d
    
    ae = Model(inp, decoded)
    ae.summary()
    print(ae.layers[19].output.shape)
    
    batch_size = 64
    ae.compile(optimizer=adam(lr=1e-4), loss='mse')
    history = ae.fit(train_data, train_data, validation_split=0.1, verbose=1, batch_size=batch_size, epochs=100)
    ae.save('ae_extractor_0714.hdf5')
    loss_plot(history.history["loss"], history.history["val_loss"])
    # Feature Extractor
    extractionLayer = 19
    f = ae.layers[extractionLayer].output
    print(f.shape)
    feature_model = Model(inputs=ae.input, outputs=f)
    return feature_model

def joint_ae(inp_dim):
    '''
    use the extracted feature from ae model as input, a basic MLP model
    '''
    clf = Sequential()
    clf.add(Dense(input_dim=inp_dim, units=64))
    clf.add(Dense(32))
    clf.add(Dense(6, activation='softmax'))
    clf.summary()
    
    clf.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
    return clf

def resnet_ldnn(input_dim):
    '''
    here we use ResNet50 to replace simple CNN block in CLDNN model
    '''
    model = Sequential()
    model.add(ResNet50(include_top=False, input_shape=(*input_dim, 1), weights=None, classes=None, pooling='average'))
    model.add(Permute((2, 1, 3)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, dropout=0.25, return_sequences=True))
    model.add(LSTM(64, dropout=0.25))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(6, kernel_regularizer=regularizers.l2(0.01), activation='softmax'))
    model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
    return model

def sfc_dcnn(input_dim):
    '''
    single feature channel CNN, use spectrogram itself as input but process time and frequency domain separately
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1, 3), padding='same', input_shape=(*input_dim, 1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(64, kernel_size=(1, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(1, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(64, kernel_size=(3, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(3, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(256, kernel_size=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(6, kernel_regularizer=regularizers.l2(0.01), activation='softmax'))
    model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
    return model

def mobilenet(input_dim):
    '''
    use MobileNet to replace simple CNN model
    '''
    model = Sequential()
    model.add(MobileNetV2(input_shape=(*input_dim, 1), alpha=1.0, include_top=False, weights=None, input_tensor=None, pooling='avg'))
    model.add(Dense(6, activation='softmax'))
    model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
    return model

def recurrent_model(input_dim):
    '''
    use spectrogram as input of RNN model, time domain first
    '''
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(*input_dim,)))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(6, kernel_regularizer=regularizers.l2(0.01), activation='softmax')) # Change
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
    return model