# from msilib.schema import Directory
from process_audio import create_mfcc, print_stats
import torchaudio
from torchaudio.transforms import Spectrogram
import os 
from util import get_project_root
import math
import json 

SAMPLE_RATE = 22050 
DURATION    = 5
JSON_PATH = "tiktok_audio_dataset.json"
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
AUDIO_FILE_PATH = '/data_collection/downloadedMp3'
DIRECTORY = str(get_project_root())+ AUDIO_FILE_PATH

JSON_DATASET = {
    'mfcc'          :  [], 
    'label'         :  [],
    'viewCount'    :  []  
}

def build_audio_dataset(num_mfcc=13, n_fft=2048, hop_length=512, num_segments=50): 
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.floor(samples_per_segment / hop_length) 
    tinku = True

    for filename in os.listdir(DIRECTORY):
        path_audio = os.path.join(DIRECTORY, filename)
        
        # print(filename)
        if str(filename).endswith('.mp3'):
            view_count = str(filename).split("-")[1]
            view_count = view_count.split(".")[0]
        else: 
            continue
        
        try: 
            waveform, sample_rate = torchaudio.load(path_audio)
        except Exception: 
            pass
        # print(len())
        if (waveform.dim() == 0): 
            print(len(mfcc))
            continue
        if len(waveform)>1: 
            waveform = waveform[0]
            
            
        for seggs in range(num_segments):   
            start = samples_per_segment * seggs
            end   = start + samples_per_segment
            mfcc_module = create_mfcc(sample_rate= sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            if (waveform.dim() == 0) or (len(waveform)<=0) or (len(waveform[start:end]) <= 0):
                print(len(mfcc))
                tinku = False 
                pass 
            else:    
                # print(len(waveform[start:end]))
                # if not 
                mfcc = mfcc_module(waveform[start:end])
                mfcc = mfcc.T

            if len(mfcc) >= num_mfcc_vectors_per_segment:
                JSON_DATASET["mfcc"].append(mfcc.tolist())
                JSON_DATASET["label"].append(0)
                JSON_DATASET["viewCount"].append(view_count)
                if not tinku: 
                    tinku = True
                    continue
            
    with open(JSON_PATH, "w") as fp:
        json.dump(JSON_DATASET, fp, indent=6)

        # print_stats(waveform, sample_rate=None, src=None)
        # print((len(waveform[0])))
        # print(SAMPLE_RATE * DURATION)
        # break 
        # print(filename)

if __name__ == "__main__":
    build_audio_dataset(num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5)
    print('JSON dataset has been built!')
