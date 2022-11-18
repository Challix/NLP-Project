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

train_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-train.csv')
val_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-val.csv')
# test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')

train_texts, train_waves, train_labels = train_dataset_.texts, train_dataset_.waves, train_dataset_.labels
val_texts, val_waves, val_labels = val_dataset_.texts, val_dataset_.waves, val_dataset_.labels
# test_texts, test_waves, test_labels = test_dataset_.texts, test_dataset_.waves, test_dataset_.labels

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)

max_len = 16000 * 3
train_audio = feature_extractor(train_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)
val_audio = feature_extractor(val_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)
# test_audio = feature_extractor(test_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)

train_dataset = SerAndTerDataset(train_encodings, train_audio, train_labels)
val_dataset = SerAndTerDataset(val_encodings, val_audio, val_labels)
# test_dataset = SerAndTerDataset(test_encodings, test_audio, test_labels)

num_labels = len(train_dataset_.targets)

text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", output_hidden_states=True, num_labels=num_labels)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er", output_hidden_states=True, num_labels=num_labels)

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

model = MLP(in_channels=1536)
model.to(device)
text_model.to(device)
audio_model.to(device)
model = DataParallel(model)
text_model = DataParallel(text_model)
audio_model = DataParallel(audio_model)

# checkpoint_path = 'results_feat_fusion-3/feat_fusion_14.pth'
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'], strict=True)
# text_model.load_state_dict(checkpoint['text_model_state_dict'], strict=True)
# audio_model.load_state_dict(checkpoint['audio_model_state_dict'], strict=True)

checkpoint_path = 'results_ter/ter_18.pth'
checkpoint = torch.load(checkpoint_path)
text_model.load_state_dict(checkpoint['text_model_state_dict'], strict=True)

checkpoint_path = 'results_ser/ser_15.pth'
checkpoint = torch.load(checkpoint_path)
audio_model.load_state_dict(checkpoint['audio_model_state_dict'], strict=True)

for param in text_model.parameters():
    param.requires_grad = True

for param in audio_model.parameters():
    param.requires_grad = True

# finetune_params = ['pre_classifier.bias', 'classifier.bias', 'pre_classifier.weight', 'classifier.weight']

# # for name, param in text_model.named_parameters():
# #     if name not in finetune_params:
# #         param.requires_grad = False
# #     else:
# #         param.requires_grad = True
# #         print(name)

# for name, param in audio_model.named_parameters():
#     if name not in finetune_params:
#         param.requires_grad = False
#     else:
#         param.requires_grad = True
#         print(name)

# params = model.parameters() + text_model.parameters() + audio_model.parameters()

optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': text_model.parameters()}, {'params': audio_model.parameters()}, ], lr=1e-4, weight_decay=1e-4)

checkpoints_path = 'results_feat_fusion-3'
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
    model.train()
    audio_model.train()
    text_model.train()

    for batch_idx, data_ in enumerate(trainloader):
        iters = epoch * len(trainloader) + batch_idx

        text_data_, data_ = data_

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
        model.eval()
        text_model.eval()
        audio_model.eval()
        pred = []
        y = []
        for ii, data_ in enumerate(validloader):
            text_data_, data_ = data_

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
        print('val f1: ', f1)

        if f1 >= BEST_F1:
            BEST_F1 = f1
            save_model({
                    'model_state_dict': model.state_dict(),
                    'text_model_state_dict': text_model.state_dict(),
                    'audio_model_state_dict': audio_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    }, checkpoints_path, 'feat_fusion', epoch+1)
            print('checkpoint saved at {}/{}'.format(epoch+1, max_epoch))
            
        logs['F1'].append(f1*100)
        logs['epochs_F1'].append(epoch)
        plotlog(logs, checkpoints_path)