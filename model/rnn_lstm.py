import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import datasets
from sklearn.model_selection import train_test_split
from util import load_data


class TikTokModel(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size, hidden_size=64, bidirectional=True, num_layers=2
        )
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, 1)
        self.fc2 = nn.Linear(44, 1)

    def forward(self, x):

        x = self.lstm1(x)
        x = self.drop(x[0])
        x = F.relu(self.fc(x)).squeeze()
        x = self.fc2(x)
        x = x.squeeze()

        return torch.sigmoid(x)


def prepare_datasets(test_size, validation_size):

    # load data
    dataset = load_data("tiktok_audio_dataset.json")
    X, y = np.array(dataset["mfcc"]), np.array(dataset["label"])
    # X = X.squeeze(1)
    print(X.shape)
    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


if __name__ == "__main__":
    # Set parameters here
    input_shape = 10
    num_units = 128

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2
    )
    input_shape = (X_train.shape[1], X_train.shape[2])

    tiktok_model = TikTokModel(X_train.shape[2])
    print(vars(tiktok_model))
