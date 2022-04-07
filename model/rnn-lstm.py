import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

# Input shape of each mfcc tensor
def build_model(input_shape, num_units): 
    layers = []
    layers.append(nn.LSTM(input_shape, bidirectional=True))
    layers.append(nn.LSTM(input_shape))
    layers.append(nn.Linear(num_units))
    layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(num_units))

    net = nn.Sequential(*layers)
    return net 

def train_model():
    # TODO: set appropriate optimizer and loss function
    return 

if __name__ == "__main__": 
    # Set parameters here 
    input_shape = 10
    num_units = 128

    rnn = build_model(input_shape, num_units)
    # train rnn 

