import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import librosa
import librosa.display
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import load_model
labels = ['female_angry', 'female_disgust', 'female_fear', 'female_happy',
 'female_neutral', 'female_sad', 'female_surprise', 'male_angry',
 'male_disgust', 'male_fear', 'male_happy', 'male_neutral', 'male_sad',
 'male_surprise']
def get_model():
  model = Sequential()
  model.add(Conv1D(256, 8, padding='same', input_shape=(1080, 1)))
  model.add(Activation('relu'))
  model.add(Conv1D(256, 8, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))
  model.add(MaxPooling1D(pool_size=(8)))
  model.add(Conv1D(128, 8, padding='same'))
  model.add(Activation('relu'))
  model.add(Conv1D(128, 8, padding='same'))
  model.add(Activation('relu'))
  model.add(Conv1D(128, 8, padding='same'))
  model.add(Activation('relu'))
  model.add(Conv1D(128, 8, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))
  model.add(MaxPooling1D(pool_size=(8)))
  model.add(Conv1D(64, 8, padding='same'))
  model.add(Activation('relu'))
  model.add(Conv1D(64, 8, padding='same'))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(14)) # Target class number
  model.add(Activation('softmax'))
  return model

def get_prediction(audio_path, model_path='model.h5'):
    data, sampling_rate = librosa.load(audio_path, res_type='kaiser_fast' ,duration=2.5 ,sr=44100 ,offset=0.5)
    sampling_rate = np.array(sampling_rate)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=13), axis=0)
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)
    X = np.array(mfcc)
    X = (X - mean) / std
    X= np.expand_dims(np.array([X]), axis=2)
    print(X.shape)
    model = load_model(model_path)
    print(model.summary())
    prediction = labels[np.argmax(model.predict(X))]
    print(prediction)
    return prediction

if __name__ == '__main__':
  get_prediction('audio.wav', 'model.h5')
