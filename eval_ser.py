import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from datasets import load_dataset, DatasetDict, Dataset, load_metric
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils import data
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, HubertForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ForSequenceClassification
import librosa
import glob
import torchaudio
import os
import soundfile as sf

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
            
            audio = librosa.load(wav_path, sr=16000, offset=start, duration=duration)

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

        # print(item)
        return item

    def __len__(self):
        return len(self.labels)

def eval_ser(dataset, model):
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
            # print(data_)

            input_values = data_['input_values']
            attention_mask = data_['attention_mask']

            predictions = model(input_values=input_values, attention_mask=attention_mask)
            print(predictions['hidden_states'][0].size())
            sys.exit()
            predictions = np.argmax(predictions['logits'], axis=1)

            y_pred.append(predictions.detach().cpu().numpy().tolist())
            y_true.append(data_['labels'].detach().cpu().numpy().tolist())


    y_pred = [item for sublist in y_pred for item in sublist]
    y_true = [item for sublist in y_true for item in sublist]

    results = compute_f1(y_pred, y_true)
    return results

test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

test_waves, test_labels = test_dataset_.waves, test_dataset_.labels

# audio file is decoded on the fly
max_len = 16000 * 3
test_audio = processor(test_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)
test_dataset = SerDataset(test_audio, test_labels)

# test finetuned model
model = Wav2Vec2ForSequenceClassification.from_pretrained("results_ser/checkpoint-14605", local_files_only=True, output_hidden_states=True)
print('model loaded')

print('test')
results = eval_ser(test_dataset, model)
print('test f1: ', results['f1'])

