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
import math
import pickle
import torch.nn as nn
from torch.nn import DataParallel
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model_dict, save_path, name, epoch):
    epoch_str = '%02d' % epoch
    save_name = os.path.join(save_path, name + '_' + epoch_str + '.pth')
    torch.save(model_dict, save_name)
    print('checkpoint %s saved' % save_name)
    return save_name

class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertImageIntermediate(nn.Module):
    def __init__(self):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertImageOutput(nn.Module):
    def __init__(self):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertImageIntermediate(nn.Module):
    def __init__(self):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertBiAttention(nn.Module):
    def __init__(self):
        super(BertBiAttention, self).__init__()
        self.visualization = False
        self.num_attention_heads = 12
        self.attention_head_size = int(
            768 / 12
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1 = nn.Linear(768, self.all_head_size)
        self.key1 = nn.Linear(768, self.all_head_size)
        self.value1 = nn.Linear(768, self.all_head_size)

        self.dropout1 = nn.Dropout(0.1)

        self.query2 = nn.Linear(768, self.all_head_size)
        self.key2 = nn.Linear(768, self.all_head_size)
        self.value2 = nn.Linear(768, self.all_head_size)

        self.dropout2 = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        # attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        # attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data = None

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data

class BertBiOutput(nn.Module):
    def __init__(self):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(768, 768)
        self.LayerNorm1 = BertLayerNorm(768, eps=1e-12)
        self.dropout1 = nn.Dropout(0.1)

        self.q_dense1 = nn.Linear(768, 768)
        self.q_dropout1 = nn.Dropout(0.1)

        self.dense2 = nn.Linear(768, 768)
        self.LayerNorm2 = BertLayerNorm(768, eps=1e-12)
        self.dropout2 = nn.Dropout(0.1)

        self.q_dense2 = nn.Linear(768, 768)
        self.q_dropout2 = nn.Dropout(0.1)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2

class BertConnectionLayer(nn.Module):
    def __init__(self):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention()

        self.biOutput = BertBiOutput()

        self.v_intermediate = BertImageIntermediate()
        self.v_output = BertImageOutput()

        self.t_intermediate = BertIntermediate()
        self.t_output = BertOutput()

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.biOutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs
        
class CoAttnMLP(nn.Module):

    def __init__(self, in_channels=1536):
        super(CoAttnMLP, self).__init__()
        self.in_channels = in_channels

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        self.coattn_layer1 = BertConnectionLayer()
        self.coattn_layer2 = BertConnectionLayer()
        self.coattn_layer3 = BertConnectionLayer()
        

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2):
        input_tensor1, input_tensor2, _ = self.coattn_layer1(input_tensor1, attention_mask1, input_tensor2, attention_mask2)
        input_tensor1, input_tensor2, _ = self.coattn_layer2(input_tensor1, attention_mask1, input_tensor2, attention_mask2)
        input_tensor1, input_tensor2, _ = self.coattn_layer3(input_tensor1, attention_mask1, input_tensor2, attention_mask2)

        x = torch.cat((input_tensor1.mean(dim=1), input_tensor2.mean(dim=1)), dim=1)

        logits = self.mlp(x)

        return logits

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

model = CoAttnMLP(in_channels=1536)
model.to(device)
text_model.to(device)
audio_model.to(device)
model = DataParallel(model)
text_model = DataParallel(text_model)
audio_model = DataParallel(audio_model)

checkpoint_path = 'results_ter/ter_18.pth'
checkpoint = torch.load(checkpoint_path)
text_model.load_state_dict(checkpoint['text_model_state_dict'], strict=True)

checkpoint_path = 'results_ser/ser_15.pth'
checkpoint = torch.load(checkpoint_path)
audio_model.load_state_dict(checkpoint['audio_model_state_dict'], strict=True)

# checkpoint_path = 'results_attn_feat_fusion-2/attn_feat_fusion_05.pth'
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'], strict=True)
# text_model.load_state_dict(checkpoint['text_model_state_dict'], strict=True)
# audio_model.load_state_dict(checkpoint['audio_model_state_dict'], strict=True)


finetune_params = ['pre_classifier.bias', 'classifier.bias', 'pre_classifier.weight', 'classifier.weight']

for param in text_model.parameters():
    param.requires_grad = True

for param in audio_model.parameters():
    param.requires_grad = True

# for name, param in audio_model.named_parameters():
#     if name not in finetune_params:
#         param.requires_grad = False
#     else:
#         param.requires_grad = True
#         # print(name)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

checkpoints_path = 'results_attn_feat_fusion'
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
max_epoch = 30

BEST_F1 = 0

for epoch in range(epoch_start, max_epoch):
    start = time.time()
    model.train()
    text_model.train()
    audio_model.train()

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

        audio_feat = audio_predictions['hidden_states'][0]
        text_feat = text_predictions['hidden_states'][0]

        label = data_['labels']
        label = label.to(device).long()

        logits = model(text_feat, text_attention_mask, audio_feat, torch.ones_like(audio_feat))

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

            audio_feat = audio_predictions['hidden_states'][0]
            text_feat = text_predictions['hidden_states'][0]

            label = data_['labels']
            label = label.to(device).long()

            logits = model(text_feat, text_attention_mask, audio_feat, audio_attention_mask)

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
                    }, checkpoints_path, 'attn_feat_fusion', epoch+1)
            print('checkpoint saved at {}/{}'.format(epoch+1, max_epoch))
            
        logs['F1'].append(f1*100)
        logs['epochs_F1'].append(epoch)
        plotlog(logs, checkpoints_path)