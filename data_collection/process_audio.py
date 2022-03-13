import torchaudio
import torch 

def mp3_to_tensor(music_id): 
    filename = 'downloadedMp3/' + music_id + '.mp3' 
    waveform, sample_rate = torchaudio.load(filename)
    return waveform, sample_rate

music_id = '6864673209724259078'
waveform, sample_rate = mp3_to_tensor(music_id)
print(waveform)
print(sample_rate)
