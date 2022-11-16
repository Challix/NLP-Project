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

def compute_metrics(eval_pred):
    metric = load_metric('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

def compute_f1(predictions, labels):
    metric = load_metric('f1')
    return metric.compute(predictions=predictions, references=labels, average='weighted', labels=list(test_dataset_.targets.values()))

bs = 64
train_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-train.csv')
val_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-val.csv')
test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')

# finetune bert on 5 emotion classes: 'xxx', 'sad', 'angry', 'fear', 'happy'
train_texts, train_labels = train_dataset_.texts, train_dataset_.labels
val_texts, val_labels = val_dataset_.texts, val_dataset_.labels
test_texts, test_labels = test_dataset_.texts, test_dataset_.labels

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = BertDataset(train_encodings, train_labels)
val_dataset = BertDataset(val_encodings, val_labels)
test_dataset = BertDataset(test_encodings, test_labels)

bs = 64
epochs = 10
lr = 5e-5
num_labels = len(train_dataset_.targets)

training_args = TrainingArguments(
    output_dir='./results_ter',          # output directory
    num_train_epochs=epochs,              # total number of training epochs
    per_device_train_batch_size=bs,  # batch size per device during training
    per_device_eval_batch_size=bs,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=lr,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# # finetune_params = ['pre_classifier.bias', 'classifier.bias', 'pre_classifier.weight', 'classifier.weight']
# # for name, param in model.named_parameters():
# #     if name not in finetune_params:
# #         param.requires_grad = False

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

# train model
trainer.train()

# get val acc.
results = trainer.evaluate()
print('val f1: ', results['eval_f1'])

# get test acc.
output = trainer.predict(test_dataset)
predictions = output.predictions
predictions = np.argmax(predictions, axis=1)
target = np.array(test_dataset.labels)
c_mat = confusion_matrix(target, predictions)
f1 = compute_f1(predictions, target)
print('test f1: ', f1['f1'])
print('test confusion matrix:', c_mat)

# test finetuned model
model = DistilBertForSequenceClassification.from_pretrained("results_ter/checkpoint-184", local_files_only=True, output_hidden_states=True)
print('model loaded')

print('test')
eval_ter(test_dataset, model)

