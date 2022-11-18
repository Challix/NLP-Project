import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from datasets import load_dataset, DatasetDict, Dataset, load_metric
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils import data
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

wav_files = glob.glob('data_processed/Session*/wav/*.wav')
wav_dict = {}
for path in wav_files:
    ID = path.split('/')[-1].split('.')[0]
    wav_dict[ID] = path

class CustomDataset(Dataset):
    def __init__(self, csv='data_processed/iemocap-text-label.csv'):
        iemo_df = pd.read_csv(csv)
        iemo_df.dropna()

        label_map = {
            'hap': 'happy',
            'exc': 'happy',
            'neu': 'neutral',
            'dis': 'sad',
            'sad': 'sad',
            'ang': 'angry',
            'fru': 'angry',
            'sur': 'xxx',
            'fea': 'xxx',
            'oth': 'xxx',
            'xxx': 'xxx',
            'sadness': 'sad',
            'joy': 'happy',
            'love': 'xxx',
            'anger': 'angry',
            'fear': 'xxx',
            'surprise': 'xxx'
        }
        
        texts = []
        labels = []
        waves = []

        targets = ['neutral', 'sad', 'angry', 'happy']
        targets_ = {
            'neutral': 0,
            'sad': 1,
            'angry': 2,
            'happy': 3,
        }
        
        for i, row in tqdm(iemo_df.iterrows()):
            # print(row)
            start = row['start_times']
            end = row['stop_times']
            duration = end - start
            
            audio = row['file_ids']

            ID = '_'.join(audio.split('_')[:-1])

            wav_path = wav_dict[ID]
            
            audio = librosa.load(wav_path, sr=16000, offset=start, duration=duration, mono=True)

            assert audio[1] == 16000
            float_audio = audio[0]

            text = row['texts']
            label = row['labels']

            if label_map[label] not in targets:
                continue

            label = label_map[label]

            labels.append(targets_[label])

            texts.append(text)
            waves.append(float_audio)

        self.texts = texts
        self.waves = waves
        self.labels = labels
        self.targets = targets_
        self.label_map = label_map

        assert len(self.texts) == len(self.labels) == len(self.waves)

    def __getitem__(self, index):
        text = self.texts[index]
        wave = self.waves[index]
        label = self.labels[index]

        return text, wave, label

    def __len__(self):
        return len(self.texts)

class TerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')

test_texts, test_waves, test_labels = test_dataset_.texts, test_dataset_.waves, test_dataset_.labels

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

test_encodings = tokenizer(test_texts, truncation=True, padding=True)

max_len = 16000 * 3

test_dataset = TerDataset(test_encodings, test_labels)

num_labels = len(test_dataset_.targets)

text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

testloader = data.DataLoader(test_dataset,
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False
                                  )

text_model.to(device)
text_model = DataParallel(text_model)

checkpoint_path = 'results_ter/ter_18.pth'
checkpoint = torch.load(checkpoint_path)
text_model.load_state_dict(checkpoint['text_model_state_dict'], strict=True)

print('Model loaded: ', checkpoint_path)

text_model.eval()
with torch.no_grad():
    pred = []
    y = []
    for batch_idx, data in enumerate(testloader):
        print('computing embeddings for batch %d/%d' % (batch_idx, len(testloader)))
        input_ids = data['input_ids']
        text_attention_mask = data['attention_mask']            

        input_ids = input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)
        
        text_predictions = text_model(input_ids=input_ids, attention_mask=text_attention_mask)

        logits = text_predictions['logits']

        label = data['labels']
        label = label.to(device).long()

        pred += np.argmax(logits.cpu().detach().numpy(), axis=1).tolist()
        y += label.cpu().detach().numpy().tolist()
    f1 = f1_score(y, pred, average='weighted')
    print('test f1: ', f1)

