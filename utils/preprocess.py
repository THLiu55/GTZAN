import json
import os
import math
import librosa
from typing import Optional
from utils.params import Params
from tqdm import tqdm
import numpy as np


def load_audio(params: Params) -> Optional[tuple]:
    data = {
        'mapping': [],
        'labels': [],
        'mfcc': []
    }
    num_samples_per_segment = int(params.sample_rate * params.track_duration)
    samples_per_segment = int(num_samples_per_segment / params.num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / params.hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(params.audio_dir)):
        if dirpath is not params.audio_dir:
            category = dirpath.split('/')[-1]
            data['mapping'].append(category)
            print(f'Processing {category}')
            for f in tqdm(filenames):
                file_path = os.path.join(dirpath, f)
                if file_path != params.ignored_audio:
                    signal, sr = librosa.load(file_path, sr=params.sample_rate)
                    for s in range(params.num_segments):
                        start_sample = samples_per_segment * s
                        finish_sample = start_sample + samples_per_segment

                        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], sr=sr, n_mfcc=params.n_mfcc, n_fft=params.n_fft, hop_length=params.hop_length)
                        mfcc = mfcc.T

                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data['mfcc'].append(mfcc.tolist())
                            data['labels'].append(i-1)
                            

    with open(params.audio_processed_path, 'w') as fp:
        print('Saving data to', params.audio_processed_path)
        json.dump(data, fp, indent=4)



def load_data(params: Params):
    print('Loading data from', params.audio_processed_path)
    with open(params.audio_processed_path, 'r') as fp:
        data = json.load(fp)
        inputs = np.array(data["mfcc"]) 
        targets = np.array(data["labels"])
    return inputs, targets
