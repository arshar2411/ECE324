from urllib import request
import pandas as pd
import os
from tqdm import tqdm

url = "tiktok-trending.csv"
tiktok_dataset = pd.read_csv(url, on_bad_lines="skip")
for i in tqdm(range(len(tiktok_dataset))):
    music_url = tiktok_dataset.loc[i, "Music URL"]
    local_file = (
        "downloadedMp3/"
        + str(tiktok_dataset.loc[i, "Music ID"])
        + "-"
        + str(tiktok_dataset.loc[i, "Play Count"])
        + ".mp3"
    )

    if (
        type(music_url) == str
        and music_url.endswith(".mp3")
        and not os.path.exists(local_file)
    ):
        try:
            request.urlretrieve(music_url, local_file)
        except Exception:
            pass
