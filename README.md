# ECE324
### Data Collection 
After you have cloned the repository, you should have `tiktok-trending.csv` dataset on your local machine inside the `/data_collection` directory. 
Some Python libraries that you will need to work with the Data Collection API are: 
- Pandas
- Playwright 
- Torch 
- TorchAudio
##### Downloading the audio files 
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
##### Using the audio files as inputs to a neural network 
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
