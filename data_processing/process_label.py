import json
from util import get_project_root, load_data, save_data
import numpy as np


def find_n_percentile(n, count):
    return np.percentile(count, n)


if __name__ == "__main__":
    json_dataset = load_data("tiktok_audio_dataset.json")
    label = np.array(json_dataset["label"]).astype(float)
    view_count = np.array(json_dataset["viewCount"]).astype(float)
    top_25_percent = find_n_percentile(75, view_count)

    for i in range(len(label)):
        if view_count[i] >= top_25_percent:
            label[i] = 1
    try:
        json_dataset["label"] = label.tolist()
        json_dataset["viewCount"] = view_count.tolist()
        save_data("tiktok_audio_dataset.json", json_dataset)
        print("json file has been updated")
    except Exception:
        print(Exception)
