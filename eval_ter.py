import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from datasets import load_dataset, DatasetDict, Dataset, load_metric
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils import data
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

        targets = ['neutral', 'sad', 'angry', 'happy']
        targets_ = {
            'neutral': 0,
            'sad': 1,
            'angry': 2,
            'happy': 3,
        }
        
        for i, row in tqdm(iemo_df.iterrows()):
            text = row['texts']
            label = row['labels']

            if label_map[label] not in targets:
                continue

            label = label_map[label]

            labels.append(targets_[label])

            texts.append(text)

        self.texts = texts
        self.labels = labels
        self.targets = targets_
        self.label_map = label_map

        assert len(self.texts) == len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        return text, label

    def __len__(self):
        return len(self.texts)

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def eval_ter(dataset, model):
    data_loader = data.DataLoader(dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=False
                                  )
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # eval on iemocap
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_idx, data_ in enumerate(data_loader):
            input_ids = data_['input_ids']
            attention_mask = data_['attention_mask']

            predictions = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = np.argmax(predictions['logits'], axis=1)

            y_pred.append(predictions.detach().cpu().numpy().tolist())
            y_true.append(data_['labels'].detach().cpu().numpy().tolist())


    y_pred = [item for sublist in y_pred for item in sublist]
    y_true = [item for sublist in y_true for item in sublist]

    results = compute_f1(y_pred, y_true)
    return results

test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')
test_texts, test_labels = test_dataset_.texts, test_dataset_.labels

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = BertDataset(test_encodings, test_labels)

# test finetuned model
model = DistilBertForSequenceClassification.from_pretrained("results_ter/checkpoint-184", local_files_only=True, output_hidden_states=True)
print('model loaded')

print('test')
res = eval_ter(test_dataset, model)
print(res)

