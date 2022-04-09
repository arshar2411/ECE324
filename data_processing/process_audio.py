import wave
import torchaudio
import torch
import matplotlib.pyplot as plt
from torchaudio.transforms import Spectrogram


def mp3_to_tensor(music_id):
    filename = "downloadedMp3/" + music_id + ".mp3"
    waveform, sample_rate = torchaudio.load(filename)
    return waveform, sample_rate


def create_spectrogram(music_id, n_fft=100):
    waveform, sample_rate = mp3_to_tensor(music_id)
    specgram = Spectrogram(n_fft)
    output = specgram(waveform)
    return output


def create_mfcc(sample_rate, n_mfcc, n_fft, hop_length):
    """
    Build mfcc for a given torchaudio waveform

    :param signal(tensor): waveform
    :param sample_rate (int) : ykik
    :param n_mfcc (int) : ykik
    :param n_fft (int) : ykik
    :param hop_length (int) : ykik
    """
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
        },
    )
    return mfcc
