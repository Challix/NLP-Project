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

from torch.nn import DataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def eval_ser_and_ter_score_fusion(dataset, text_model, audio_model):
    data_loader = data.DataLoader(dataset,
                                  batch_size=64,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False
                                  )

    # eval on iemocap
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_idx, data_ in enumerate(data_loader):

            text_data_, data_ = data_

            input_ids = text_data_['input_ids']
            text_attention_mask = text_data_['attention_mask']            

            input_values = data_['input_values']
            audio_attention_mask = data_['attention_mask']

            audio_predictions = audio_model(input_values=input_values, attention_mask=audio_attention_mask)
            
            text_predictions = text_model(input_ids=input_ids, attention_mask=text_attention_mask)

            predictions = 1/2*audio_predictions['logits'] + 1/2*text_predictions['logits']
            predictions = np.argmax(predictions.cpu().detach().numpy(), axis=1)

            y_pred.append(predictions.tolist())
            y_true.append(data_['labels'].cpu().detach().numpy().tolist())

    y_pred = [item for sublist in y_pred for item in sublist]
    y_true = [item for sublist in y_true for item in sublist]

    results = compute_f1(y_pred, y_true)
    return results

def compute_metrics(eval_pred):
    metric = load_metric('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

def compute_f1(predictions, labels):
    metric = load_metric('f1')
    return metric.compute(predictions=predictions, references=labels, average='weighted', labels=list(test_dataset_.targets.values()))

test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')

test_texts, test_waves, test_labels = test_dataset_.texts, test_dataset_.waves, test_dataset_.labels

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

test_encodings = tokenizer(test_texts, truncation=True, padding=True)

max_len = 16000 * 3
test_audio = feature_extractor(test_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)

test_dataset = SerAndTerDataset(test_encodings, test_audio, test_labels)

num_labels = len(test_dataset_.targets)

text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", output_hidden_states=True, num_labels=num_labels)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er", output_hidden_states=True, num_labels=num_labels)

text_model.to(device)
audio_model.to(device)

text_model = DataParallel(text_model)
audio_model = DataParallel(audio_model)

checkpoint_path = 'results_ter/ter_18.pth'
checkpoint = torch.load(checkpoint_path)
text_model.load_state_dict(checkpoint['text_model_state_dict'], strict=True)

checkpoint_path = 'results_ser/ser_15.pth'
checkpoint = torch.load(checkpoint_path)
audio_model.load_state_dict(checkpoint['audio_model_state_dict'], strict=True)

res = eval_ser_and_ter_score_fusion(test_dataset, text_model, audio_model)
print('score fusion f1: ', res['f1'])



