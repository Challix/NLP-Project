import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow_text as text

from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from tensorflow.keras.utils import Sequence
from math import ceil

from sklearn.metrics import confusion_matrix
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from time import time

import librosa
import pandas as pd
import glob
import random
import sys

def load_cremad(percent_train):
    # TODO: Change below to your crema path
    wav_files = glob.glob('/home/groszste/Documents/CSE_842/CREMA-D/AudioWAV/*')

    sentence_mapping = {
        'IEO':'It\'s eleven o\'clock',
        'TIE':'That is exactly what happened',
        'IOM':'I\'m on my way to the meeting',
        'IWW':'I wonder what this is about',
        'TAI':'The airplane is almost full',
        'MTI':'Maybe tomorrow it will be cold',
        'IWL':'I would like a new alarm clock',
        'ITH':'I think I have a doctor\'s appointment',
        'DFA':'Don\'t forget a jacket',
        'ITS':'I think I\'ve seen this before',
        'TSI':'The surface is slick',
        'WSI':'We\'ll stop in a couple of minutes'
    }

    # filepath, speaker, sentence, emotion, intensity
    data_raw = {
        'filepath' : [],
        'speaker' : [],
        'sentence_short' : [],
        'sentence_text' : [],
        'emotion' : [],
        'intensity' : [],
        'waveform' : []
    }

    emotions_mapping = {'ANG': 0, 'HAP': 1, 'SAD': 2, 'NEU': 3}

    max_n = 0
    for wav_file in wav_files:
        speaker, sentence, emotion, intensity = wav_file.split('/')[-1].split('.')[0].split('_')

        if emotion not in list(emotions_mapping.keys()):
            continue

        data_raw['filepath'].append(wav_file)
        data_raw['speaker'].append(speaker)
        data_raw['sentence_short'].append(sentence)
        data_raw['sentence_text'].append(sentence_mapping[sentence])
        data_raw['emotion'].append(emotions_mapping[emotion])
        data_raw['intensity'].append(intensity)

        wav_data, sr = librosa.load(wav_file, sr=16000, mono=True)
        wav_data = list(wav_data)

        if len(wav_data) > N_sequence:
            wav_data = wav_data[:N_sequence]
        else:
            wav_data += [0.0] * (N_sequence - len(wav_data))

        if len(wav_data) > max_n:
            max_n = len(wav_data)

        data_raw['waveform'].append(wav_data)

    c = list(zip(data_raw['waveform'], data_raw['emotion'], data_raw['sentence_text']))
    random.shuffle(c)
    waveforms, emotions, sentences = zip(*c)

    waveforms = np.array(waveforms)
    emotions = np.array(emotions)
    sentences = np.array(sentences)

    n_train = int(percent_train * len(waveforms))
    n_test = int((1-percent_train)/2 * len(waveforms))
    n_val = len(waveforms) - n_train - n_test
    
    waveforms_train, emotions_train, sentences_train = waveforms[:n_train], emotions[:n_train], sentences[:n_train]
    waveforms_val, emotions_val, sentences_val = waveforms[n_train:n_train+n_val], emotions[n_train:n_train+n_val], sentences[n_train:n_train+n_val]
    waveforms_test, emotions_test, sentences_test = waveforms[n_train+n_val:], emotions[n_train+n_val:], sentences[n_train+n_val:]

    data_train = (waveforms_train, emotions_train, sentences_train)
    data_val = (waveforms_val, emotions_val, sentences_val)
    data_test = (waveforms_test, emotions_test, sentences_test)
    
    return data_train, data_val, data_test

def build_finetune_model(base_model, num_classes):
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

BATCH_SIZE = 2
NUM_EPOCHS = 1
PERCENT_TRAIN = 0.8
N_sequence = 1000
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = Adam(lr=0.0001)

data_train, data_val, data_test = load_cremad(PERCENT_TRAIN)

# Prepare the audio training dataset.
X_train_audio = data_train[0].tolist()
X_val_audio = data_val[0].tolist()
X_test_audio = data_test[0].tolist()

Y_train_audio = data_train[1].tolist()
Y_val_audio = data_val[1].tolist()
Y_test_audio = data_test[1].tolist()

# Prepare the train dataset.
print("creating audio dataset")
train_dataset_audio = tf.data.Dataset.from_tensor_slices((X_train_audio, Y_train_audio))
train_dataset_audio = train_dataset_audio.shuffle(buffer_size=1024).batch(BATCH_SIZE)
# Prepare the validation dataset.
val_dataset_audio = tf.data.Dataset.from_tensor_slices((X_val_audio, Y_val_audio))
val_dataset_audio = val_dataset_audio.batch(BATCH_SIZE)
# Prepare the validation dataset.
test_dataset_audio = tf.data.Dataset.from_tensor_slices((X_test_audio, Y_test_audio))
test_dataset_audio = test_dataset_audio.batch(BATCH_SIZE)

print('building audio model')
trill_model = tf.keras.models.Sequential()
trill_model.add(keras.layers.Input(batch_shape=(None, N_sequence)))  # Input is [bs, input_length]
trill_layer = hub.KerasLayer(
    handle='https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3',
    trainable=False,
    arguments={'sample_rate': tf.constant(16000, tf.int32)},
    output_key='embedding',
    output_shape=[None,2048]
)
trill_model.add(trill_layer)
audio_model = build_finetune_model(trill_model, 4)

audio_model.compile(optimizer, loss=loss_fn, metrics=['accuracy'])
audio_model.summary()

print('fitting audio model')
audio_model.fit(train_dataset_audio, validation_data=val_dataset_audio, epochs=NUM_EPOCHS)

score = audio_model.evaluate(X_test_audio, Y_test_audio, verbose = 0) 
print('Audio Model Test loss:', score[0]) 
print('Audio Model Test accuracy:', score[1])


print("creating text dataset")
X_train_text = data_train[2].tolist()
X_val_text = data_val[2].tolist()
X_test_text = data_test[2].tolist()

Y_train_text = data_train[1].tolist()
Y_val_text = data_val[1].tolist()
Y_test_text = data_test[1].tolist()

train_dataset_text = tf.data.Dataset.from_tensor_slices((X_train_text, Y_train_text))
train_dataset_text = train_dataset_text.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# Prepare the validation dataset.
val_dataset_text = tf.data.Dataset.from_tensor_slices((X_val_text, Y_val_text))
val_dataset_text = val_dataset_text.batch(BATCH_SIZE)
# Prepare the validation dataset.
test_dataset_text = tf.data.Dataset.from_tensor_slices((X_test_text, Y_test_text))
test_dataset_text = test_dataset_text.batch(BATCH_SIZE)

print('building text model')
#build bert model
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/4",
    trainable=True)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]   
bert_model = tf.keras.Model(text_input, pooled_output)
text_model = build_finetune_model(bert_model, 4)

text_model.compile(optimizer, loss=loss_fn, metrics=['accuracy'])
text_model.summary()

print('fitting text model')
text_model.fit(train_dataset_text, validation_data=val_dataset_text, epochs=NUM_EPOCHS)
score = text_model.evaluate(X_test_text, Y_test_text, verbose = 0) 

print('Text Model Test loss:', score[0]) 
print('Text Model Test accuracy:', score[1])