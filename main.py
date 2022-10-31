import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()
import tensorflow_hub as hub
from tensorflow import keras

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

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

# set up GPU's
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 2
NUM_EPOCHS = 10
PERCENT_TRAIN = 0.8
# N_sequence = 80000
N_sequence = 1000
model_name = 'trill'

def load_cremad(percent_train):
    # TODO: Change below to your crema path
    wav_files = glob.glob('../../data/AudioWAV/*')

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

    # New softmax layer
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

data_train, data_val, data_test = load_cremad(PERCENT_TRAIN)

num_train_images = 0
num_val_images = 0

trill_model = tf.keras.models.Sequential()
trill_model.add(tf.keras.Input((N_sequence,)))  # Input is [bs, input_length]
trill_layer = hub.KerasLayer(
    handle='https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3',
    trainable=True,
    arguments={'sample_rate': tf.constant(16000, tf.int32)},
    output_key='embedding',
    output_shape=[None, 2048]
)
trill_model.add(trill_layer)

assert trill_layer.trainable_variables

model = build_finetune_model(trill_model, 4)

# #test model
# wav_as_float_or_int16 = np.sin(np.linspace(-np.pi, np.pi, N_sequence), dtype=np.float32)
# inp = np.vstack((wav_as_float_or_int16,wav_as_float_or_int16))
# emb = model(inp)
# print(emb)

optimizer = Adam(lr=0.0001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
X_train = data_train[0].tolist()
X_val = data_val[0].tolist()
X_test = data_test[0].tolist()

Y_train = data_train[1].tolist()
Y_val = data_val[1].tolist()
Y_test = data_test[1].tolist()

# Prepare the train dataset.
print("creating train dataset")
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

# Prepare the validation dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)  # Logits for this minibatch
            loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))

model.save("saved_models")
