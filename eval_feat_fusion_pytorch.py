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

class MLP(nn.Module):

    def __init__(self, in_channels=1536):
        super(MLP, self).__init__()
        self.in_channels = in_channels

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.mlp(x)

        return logits

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

class SerAndTerDataset(torch.utils.data.Dataset):
    def __init__(self, text_encodings, wave_encodings, labels):
        self.text_encodings = text_encodings
        self.wave_encodings = wave_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.text_encodings.items()}
        item_wave = {key: torch.tensor(val[idx]) for key, val in self.wave_encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item_wave['labels'] = torch.tensor(self.labels[idx])
        return item, item_wave

    def __len__(self):
        return len(self.labels)

test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')

test_texts, test_waves, test_labels = test_dataset_.texts, test_dataset_.waves, test_dataset_.labels

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

test_encodings = tokenizer(test_texts, truncation=True, padding=True)

max_len = 16000 * 3
test_audio = feature_extractor(test_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)

test_dataset = SerAndTerDataset(test_encodings, test_audio, test_labels)

num_labels = len(test_dataset_.targets)

text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", output_hidden_states=True, num_labels=num_labels)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er", output_hidden_states=True, num_labels=num_labels)

testloader = data.DataLoader(test_dataset,
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False
                                  )

model = MLP(in_channels=1536)
model.to(device)
text_model.to(device)
audio_model.to(device)
model = DataParallel(model)
text_model = DataParallel(text_model)
audio_model = DataParallel(audio_model)

checkpoint_path = 'results_feat_fusion/feat_fusion_04.pth'

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
text_model.load_state_dict(checkpoint['text_model_state_dict'], strict=True)
audio_model.load_state_dict(checkpoint['audio_model_state_dict'], strict=True)

print('Model loaded: ', checkpoint_path)

model.eval()
text_model.eval()
audio_model.eval()
with torch.no_grad():
    pred = []
    y = []
    for batch_idx, data in enumerate(testloader):
        print('computing embeddings for batch %d/%d' % (batch_idx, len(testloader)))

        text_data_, data_ = data

        input_ids = text_data_['input_ids']
        text_attention_mask = text_data_['attention_mask']            

        input_values = data_['input_values']
        audio_attention_mask = data_['attention_mask']

        input_ids = input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)

        input_values = input_values.to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        
        audio_predictions = audio_model(input_values=input_values, attention_mask=audio_attention_mask)
        
        text_predictions = text_model(input_ids=input_ids, attention_mask=text_attention_mask)

        audio_feat = audio_predictions['hidden_states'][0].mean(dim=1)
        text_feat = text_predictions['hidden_states'][0].mean(dim=1)

        concat_out_feat = torch.cat((audio_feat, text_feat), dim=1)

        label = data_['labels']
        label = label.to(device).long()

        logits = model(concat_out_feat)

        pred += np.argmax(logits.cpu().detach().numpy(), axis=1).tolist()
        y += label.cpu().detach().numpy().tolist()
    f1 = f1_score(y, pred, average='weighted')
    print('test f1: ', f1)

