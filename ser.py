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

# wav_files = glob.glob('data/Session*/wav/*.wav')
# wav_dict = {}
# for path in tqdm(wav_files):

#     outpath = path.replace('data/', 'data_processed/')
#     audio = librosa.load(path, sr=48000)
#     float_audio = librosa.core.resample(
#           audio[0], orig_sr=48000, target_sr=16000, 
#           res_type='kaiser_best')

#     outd = '/'.join(outpath.split('/')[:-1])
#     if not os.path.isdir(outd):
#         os.makedirs(outd)

#     sf.write(outpath, float_audio, 16000)
# sys.exit()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# iemo_df = pd.read_csv('data/iemocap-text-label.csv')
# iemo_df = iemo_df.sample(frac=1, random_state=42)

# n_train = int(0.8 * len(iemo_df))
# n_test = int(0.1 * len(iemo_df))

# train = iemo_df[:n_train]
# test = iemo_df[n_train:n_train+n_test]
# validate = iemo_df[n_train+n_test:]

# train.to_csv(path_or_buf='data/iemocap-text-label-train.csv', sep=',')
# validate.to_csv(path_or_buf='data/iemocap-text-label-val.csv', sep=',')
# test.to_csv(path_or_buf='data/iemocap-text-label-test.csv', sep=',')

# sys.exit()

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

def compute_metrics(eval_pred):
    metric = load_metric('f1')
    predictions, labels = eval_pred
    predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

def compute_f1(predictions, labels):
    metric = load_metric('f1')
    return metric.compute(predictions=predictions, references=labels, average='weighted', labels=list(test_dataset_.targets.values()))

def eval_ser(dataset, model):
    data_loader = data.DataLoader(dataset,
                                  batch_size=64,
                                  shuffle=True,
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

            predictions = np.argmax(predictions['logits'], axis=1)

            y_pred.append(predictions.detach().cpu().numpy().tolist())
            y_true.append(data_['labels'].detach().cpu().numpy().tolist())


    y_pred = [item for sublist in y_pred for item in sublist]
    y_true = [item for sublist in y_true for item in sublist]

    results = compute_f1(y_pred, y_true)
    return results

train_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-train.csv')
val_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-val.csv')
test_dataset_ = CustomDataset(csv='data_processed/iemocap-text-label-test.csv')

# test ser
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

train_waves, train_labels = train_dataset_.waves, train_dataset_.labels
val_waves, val_labels = val_dataset_.waves, val_dataset_.labels
test_waves, test_labels = test_dataset_.waves, test_dataset_.labels

# audio file is decoded on the fly
max_len = 16000 * 3
train_audio = processor(train_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)
val_audio = processor(val_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)
test_audio = processor(test_waves, sampling_rate=16000, padding=True, truncation=True, max_length=max_len)


train_dataset = SerDataset(train_audio, train_labels)
val_dataset = SerDataset(val_audio, val_labels)
test_dataset = SerDataset(test_audio, test_labels)

bs = 8
epochs = 5
lr = 5e-5
num_labels = len(train_dataset_.targets)

# training_args = TrainingArguments(
#     output_dir='./results_ser',          # output directory
#     num_train_epochs=epochs,              # total number of training epochs
#     per_device_train_batch_size=bs,  # batch size per device during training
#     # per_device_eval_batch_size=2,   # batch size for evaluation
#     warmup_steps=0,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     logging_steps=10,
#     # load_best_model_at_end=True,
#     # metric_for_best_model='f1',
#     # evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=lr,
# )

# model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er", num_labels=num_labels)

# # finetune_params = ['pre_classifier.bias', 'classifier.bias', 'pre_classifier.weight', 'classifier.weight']
# # for name, param in model.named_parameters():
# #     if name not in finetune_params:
# #         param.requires_grad = False

# trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     # eval_dataset=val_dataset,             # evaluation dataset
#     # compute_metrics=compute_metrics
# )

# trainer.train()

# results = trainer.evaluate()
# print('val f1: ', results['eval_f1'])

# output = trainer.predict(test_dataset)
# predictions = output.predictions
# predictions = np.argmax(predictions, axis=1)
# target = np.array(test_dataset.labels)
# c_mat = confusion_matrix(target, predictions)
# f1 = compute_f1(predictions, target)
# print('test f1: ', f1['f1'])
# print('test confusion matrix:', c_mat)

# test finetuned model
model = Wav2Vec2ForSequenceClassification.from_pretrained("results_ser/checkpoint-14605", local_files_only=True, output_hidden_states=True)
print('model loaded')

print('test')
results = eval_ser(test_dataset, model)
print('test f1: ', results['f1'])

