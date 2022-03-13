import torchaudio
import torch 

def mp3_to_tensor(music_id): 
    filename = 'downloadedMp3/' + music_id + '.mp3' 
    waveform, sample_rate = torchaudio.load(filename)
    return waveform, sample_rate
