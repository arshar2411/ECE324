import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import datasets
from sklearn.model_selection import train_test_split
from util import load_data

# Input shape of each mfcc tensor
def build_model(input_size, hidden_size): 
    layers = []
    # nn.LSTM()
    layers.append(nn.LSTM(input_size = input_size, hidden_size=hidden_size, bidirectional=True, num_layers=2))
    # layers.append(nn.LSTM(num_units))
    # layers.append(nn.Linear(num_units))
    # layers.append(nn.Dropout(0.2))
    # layers.append(nn.Linear(num_units))

    net = nn.Sequential(*layers)
    return net 

class TikTokModel(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size = input_size, hidden_size=64, bidirectional=True, num_layers=2)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, 1)
        self.fc2 = nn.Linear(44,1)

        # self.fc2 = nn.Linear(44, 1)
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):

        x = self.lstm1(x)
        # print(len(x[1]))
        # print(x[0].shape)
        x = self.drop(x[0])
        x = F.relu(self.fc(x)).squeeze()
        x = self.fc2(x)

        # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        # print(x.flatten)
        return torch.sigmoid(x)
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)

def prepare_datasets(test_size, validation_size):
  
    # load data
    dataset = load_data('tiktok_audio_dataset.json')
    X, y = np.array(dataset['mfcc']), np.array(dataset['label'])
    # X = X.squeeze(1)
    print(X.shape)
    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def train_model():
    # TODO: set appropriate optimizer and loss function
    return 

if __name__ == "__main__": 
    # Set parameters here 
    input_shape = 10
    num_units = 128

    # rnn = build_model(input_shape, num_units)
    # train rnn 

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    print(len(X_validation[0]))
    print(X_train.shape)
    print(X_validation.shape)
    input_shape = (X_train.shape[1], X_train.shape[2])
    # rnn_model =  build_model(X_train.shape[2], 64)

    tiktok_model = TikTokModel(X_train.shape[2])
    print(vars(tiktok_model))

