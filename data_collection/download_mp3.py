from urllib import request
import pandas as pd

url="tiktok-trending.csv"
tiktok_dataset=pd.read_csv(url, on_bad_lines='skip')
for i in range(len(tiktok_dataset)) :
  music_url = tiktok_dataset.loc[i, "Music URL"]
  if music_url.endswith('.mp3'): 
      remote_url = music_url
      local_file = 'downloadedMp3/' + str(tiktok_dataset.loc[i, "Music ID"]) + '.mp3'
      request.urlretrieve(remote_url, local_file)
