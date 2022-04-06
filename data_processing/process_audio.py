import wave
import torchaudio
import torch
import matplotlib.pyplot as plt
from torchaudio.transforms import Spectrogram
 
def mp3_to_tensor(music_id):
   filename = 'downloadedMp3/' + music_id + '.mp3'
   waveform, sample_rate = torchaudio.load(filename)
   return waveform, sample_rate
 
def create_spectrogram(music_id, n_fft=100):
   waveform, sample_rate = mp3_to_tensor(music_id)
   specgram = Spectrogram(n_fft)
   output = specgram(waveform)
   return output


def create_mfcc(music_id, n_fft=100):
  waveform, sample_rate = mp3_to_tensor(music_id)
  n_fft = 400.0
  frame_length = n_fft / sample_rate * 1000.0
  frame_shift = frame_length / 2.0

  params = {
    "channel": 0,
    "dither": 0.0,
    "window_type": "hanning",
    "frame_length": frame_length,
    "frame_shift": frame_shift,
    "remove_dc_offset": False,
    "round_to_power_of_two": False,
    "sample_frequency": sample_rate,
  }
  mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)
  return mfcc 

def print_stats(waveform, sample_rate=None, src=None):
 if src:
   print("-" * 10)
   print("Source:", src)
   print("-" * 10)
 if sample_rate:
   print("Sample Rate:", sample_rate)
 print("Shape:", tuple(waveform.shape))
 print("Dtype:", waveform.dtype)
 print(f" - Max:     {waveform.max().item():6.3f}")
 print(f" - Min:     {waveform.min().item():6.3f}")
 print(f" - Mean:    {waveform.mean().item():6.3f}")
 print(f" - Std Dev: {waveform.std().item():6.3f}")
 print()
 print(waveform)
 print()
