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

train_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-train.csv')
val_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-val.csv')
# test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')

train_texts, train_waves, train_labels = train_dataset_.texts, train_dataset_.waves, train_dataset_.labels
val_texts, val_waves, val_labels = val_dataset_.texts, val_dataset_.waves, val_dataset_.labels
# test_texts, test_waves, test_labels = test_dataset_.texts, test_dataset_.waves, test_dataset_.labels

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

max_len = 16000 * 3
train_audio = feature_extractor(train_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)
val_audio = feature_extractor(val_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)
# test_audio = feature_extractor(test_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)

train_dataset = SerDataset(train_audio, train_labels)
val_dataset = SerDataset(val_audio, val_labels)
# test_dataset = SerDataset(test_audio, test_labels)

num_labels = len(train_dataset_.targets)

audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er", output_hidden_states=False)

trainloader = data.DataLoader(train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True
                                  )
validloader = data.DataLoader(val_dataset,
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False
                                  )

criterion = torch.nn.CrossEntropyLoss()

audio_model.to(device)
audio_model = DataParallel(audio_model)

for param in audio_model.parameters():
    param.requires_grad = True

# finetune_params = ['module.projector.bias', 'module.classifier.bias', 'module.projector.weight', 'module.classifier.weight']
# for name, param in audio_model.named_parameters():
#     if name not in finetune_params:
#         print(name)
#         param.requires_grad = True
#     else:
#         print(name)
#         param.requires_grad = True

optimizer = torch.optim.AdamW(audio_model.parameters(), lr=1e-4, weight_decay=1e-4)

checkpoints_path = 'results_ser-2'
if not os.path.isdir(checkpoints_path):
    os.makedirs(checkpoints_path)

epoch_start = 0
logs = {}
logs['iters_loss'] = []
logs['epochs_F1'] = []
logs['loss'] = []
logs['F1'] = []
logs['messages'] = []

print_freq = 100 
max_epoch = 20

BEST_F1 = 0

for epoch in range(epoch_start, max_epoch):
    start = time.time()
    audio_model.train()

    for batch_idx, data_ in enumerate(trainloader):
        iters = epoch * len(trainloader) + batch_idx

        input_values = data_['input_values']
        audio_attention_mask = data_['attention_mask']

        input_values = input_values.to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        
        audio_predictions = audio_model(input_values=input_values, attention_mask=audio_attention_mask)
        
        logits = audio_predictions['logits']

        label = data_['labels']
        label = label.to(device).long()

        loss = criterion(logits, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iters % print_freq == 0:
            pred_y = np.argmax(logits.cpu().detach().numpy(), axis=1)
            f1 = f1_score(label.cpu().detach().numpy(), pred_y, average='weighted')
            speed = (time.time() - start) / print_freq
            time_str = time.asctime(time.localtime(time.time()))
            try:
                iteration_message = '{} train epoch {}/{}, iter {}/{}, {:.2f} s/iter, loss {}, f1 {}'.format(time_str, epoch+1, max_epoch, batch_idx, len(trainloader), speed, loss.item(), f1)
            except:
                iteration_message = '{} train epoch {}/{}, iter {}/{}, {:.2f} s/iter, loss {}, f1 {}'.format(time_str, epoch+1, max_epoch, batch_idx, len(trainloader), speed, loss.item(), f1)
            print(iteration_message)
            logs['messages'].append(iteration_message)
            start = time.time()
            logs['iters_loss'].append(iters)
            logs['loss'].append(loss.data.cpu().numpy())

    with torch.no_grad():
        audio_model.eval()
        pred = []
        y = []
        for ii, data_ in enumerate(validloader):
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
        print('val f1: ', f1)

        if f1 >= BEST_F1:
            BEST_F1 = f1
            save_model({
                    'audio_model_state_dict': audio_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    }, checkpoints_path, 'ser', epoch+1)
            print('checkpoint saved at {}/{}'.format(epoch+1, max_epoch))
            
        logs['F1'].append(f1*100)
        logs['epochs_F1'].append(epoch)
        plotlog(logs, checkpoints_path)