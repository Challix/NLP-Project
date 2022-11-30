import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from datasets import load_dataset, DatasetDict, Dataset, load_metric
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils import data
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import os
import glob
import librosa
import pickle
import torch.nn as nn
from torch.nn import DataParallel
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plotlog(logs, resultlog_path):
    plt.figure()
    plt.plot(logs['iters_loss'], logs['loss'], linewidth=2.0)
    plt.title("Training loss")
    plt.xlabel("Iteration")
    plt.grid()
    plt.savefig(resultlog_path + '/Trainingloss.png')
    plt.clf()

    plt.figure()
    plt.plot(logs['epochs_F1'], logs['F1'], linewidth=2.0)
    plt.title("Val F1")
    plt.xlabel("epoch")
    plt.grid()
    plt.savefig(resultlog_path + '/ValidationF1.png')
    plt.clf()

def save_model(model_dict, save_path, name, epoch):
    epoch_str = '%02d' % epoch
    save_name = os.path.join(save_path, name + '_' + epoch_str + '.pth')
    torch.save(model_dict, save_name)
    print('checkpoint %s saved' % save_name)
    return save_name

wav_files = glob.glob('data_processed/Session*/wav/*.wav')
wav_dict = {}
for path in wav_files:
    ID = path.split('/')[-1].split('.')[0]
    wav_dict[ID] = path

class CustomDataset(Dataset):
    def __init__(self, csv='data_processed/iemocap-text-label.csv'):
        wav_files = glob.glob('../../data/AudioWAV/*')

        targets = ['neutral', 'sad', 'angry', 'happy']
        targets_ = {
            'neutral': 0,
            'sad': 1,
            'angry': 2,
            'happy': 3,
        }

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
        label_map = {
            'HAP': 'happy',
            'NEU': 'neutral',
            'DIS': 'sad',
            'ANG': 'angry',
            'FEA': 'xxx',
            'SAD': 'sad'
        }

        for wav_file in wav_files:
            speaker, sentence, emotion, intensity = wav_file.split('/')[-1].split('.')[0].split('_')
                    
            if label_map[emotion] not in targets:
                continue

            label = label_map[emotion]

            data_raw['filepath'].append(wav_file)
            data_raw['speaker'].append(speaker)
            data_raw['sentence_short'].append(sentence)
            data_raw['sentence_text'].append(sentence_mapping[sentence])
            
            data_raw['emotion'].append(targets_[label])
            data_raw['intensity'].append(intensity)

            wav_data, sr = librosa.load(wav_file, sr=16000, mono=True)
            data_raw['waveform'].append(wav_data)

        data_df = pd.DataFrame(data_raw)
        #print(data_df.head(10))

        self.texts = data_df['sentence_text'].to_numpy()
        self.waves = data_df['waveform'].to_numpy()
        self.labels = data_df['emotion'].to_numpy()
        self.targets = targets_

        assert len(self.texts) == len(self.labels) == len(self.waves)

    def __getitem__(self, index):
        text = self.texts[index]
        wave = self.waves[index]
        label = self.labels[index]

        return text, wave, label

    def __len__(self):
        return len(self.texts)

class SerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset_ = CustomDataset()

test_texts, test_waves, test_labels = test_dataset_.texts, test_dataset_.waves, test_dataset_.labels

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

max_len = 16000 * 3
test_audio = feature_extractor(test_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)

test_dataset = SerDataset(test_audio, test_labels)

num_labels = len(test_dataset_.targets)

audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er", output_hidden_states=True, num_labels=num_labels)

testloader = data.DataLoader(test_dataset,
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False
                                  )

audio_model.to(device)
audio_model = DataParallel(audio_model)

checkpoint_path = 'results_ser/ser_15.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
audio_model.load_state_dict(checkpoint['audio_model_state_dict'], strict=True)

print('Model loaded: ', checkpoint_path)

audio_model.eval()
with torch.no_grad():
    pred = []
    y = []
    for batch_idx, data_ in enumerate(testloader):
        print('computing embeddings for batch %d/%d' % (batch_idx, len(testloader)))
        input_values = data_['input_values']
        audio_attention_mask = data_['attention_mask']

        input_values = input_values.to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        
        audio_predictions = audio_model(input_values=input_values, attention_mask=audio_attention_mask)

        logits = audio_predictions['logits']

        label = data_['labels']
        label = label.to(device).long()

        pred += np.argmax(logits.cpu().detach().numpy(), axis=1).tolist()
        y += label.cpu().detach().numpy().tolist()
    f1 = f1_score(y, pred, average='weighted')
    print('test f1: ', f1)

