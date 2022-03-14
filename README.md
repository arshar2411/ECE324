# TikTok Sound Popularity Predictor 

![We all love ECE 324 <3](https://i.ytimg.com/vi/E71FfhDpOQ8/maxresdefault.jpg)
### Data Collection 
After you have cloned the repository, you should have `tiktok-trending.csv` dataset on your local machine inside the `/data_collection` directory. 
Some Python libraries that you will need to work with the Data Collection API are: 
- Pandas
- Playwright 
- Torch 
- TorchAudio
#### Downloading the audio files 
While you are inside the `/data_collection` directory, create a new directory `/downloadedMp3`. 
> Its important that you have the exact same spelling. 
Run the follwing command from `/data_collection`: 
```
python3 download_mp3.py
```
Now you will have all the TikTok audio files on your computer stored in the directory, `/data_collection/downloadedMp3`. Your directory should look like:
```
ECE324
│   README.md
│   LICENSE   
│
└───data_collection
    │   collect_data.py
    │   collect_data.py
    │   process_audio.py
    │   tiktok-trending.csv
    │   
    └───downloadedMP3
        │   music_id1.mp3
        │   music_id2.mp3
        │   ...
   

```
#### Using the audio files as inputs to a neural network 
The audio files (in mp3 format) can be transformed into Pytorch tensors which would make them appropriate for use in Pytorch neural networks. 
Every audio file has a unique ID which can be seen from `tiktok-trending.csv` dataset. (This is the Music ID column)
To transform the audio file into a Pytorch Tensor which represents its waveform, 
- Import the necessary function 
```python3
from process_audio import mp3_to_tensor
```
- Pass in the unique Music ID associated with each audio file and a tuple consisting of the waveform(Pytorch Tensor) and sample rate will be returned. 
```python3
waveform, sample_rate = mp3_to_tensor(music_id)
```
### Audio Processing
Audio files are converted into *spectrograms* before it is used in the neural network. Spectrogram is a visual represention of the audio file which is a spectrum of frequencies of a signal as it varies with time. 
![Say no to abortions](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_002.png)
This is done by using our audio processing API which is available under `/data_processing`. 
- Waveform is first created by running `mp3_to_tensor(music_id)` on an audio file. 
- The waveform returned is used to generate the spectrogram by running `create_spectrogram(music_id, n_fft=100)`. 

