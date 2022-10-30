import librosa
import numpy as np
import pandas as pd
import glob

# TODO: Change below to your crema path
wav_files = glob.glob('../../data/AudioWAV/*')

sentence_mapping = {
    'IEO':'It\'s eleven o\'clock',
    'TIE':'That is exactly what happened',
    'IOM':'I\'m on my way to the meeting',
    'IWW':'I wonder what this is about',
    'TAI':'The airplane is almost full',
    'MTI':'Maybe tomorrow it will be cold',
    'IWL':'I would like a new alarm clock',
    'ITH':'I think I have a doctor\'s appointment',
    'DFA':'Don\'t forget a jacket',
    'ITS':'I think I\'ve seen this before',
    'TSI':'The surface is slick',
    'WSI':'We\'ll stop in a couple of minutes'
}

# filepath, speaker, sentence, emotion, intensity
data_raw = {
    'filepath' : [],
    'speaker' : [],
    'sentence_short' : [],
    'sentence_text' : [],
    'emotion' : [],
    'intensity' : [],
    'waveform' : []
}

for wav_file in wav_files:
    speaker, sentence, emotion, intensity = wav_file.split('/')[-1].split('.')[0].split('_')
    data_raw['filepath'].append(wav_file)
    data_raw['speaker'].append(speaker)
    data_raw['sentence_short'].append(sentence)
    data_raw['sentence_text'].append(sentence_mapping[sentence])
    data_raw['emotion'].append(emotion)
    data_raw['intensity'].append(intensity)

    wav_data, sr = librosa.load(wav_file, sr=16000, mono=True)
    data_raw['waveform'].append(wav_data)

data_df = pd.DataFrame(data_raw)
print(data_df.head(10))
