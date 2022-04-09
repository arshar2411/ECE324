import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import datasets
from sklearn.model_selection import train_test_split
from util import load_data
from rnn_lstm import TikTokModel, prepare_datasets
from tqdm.auto import tqdm

def set_hyperparameters(): 
    learning_rate = 0.001
    epochs = 1000

    return learning_rate, epochs
def set_loss_and_optimizer(model, learning_rate): 
    return nn.BCELoss(), torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_model(optimizer, criterion, X_train, y_train, model, epochs ): 
    losses = []
    # Iterations = []
    # Accuracy = []
    # iter = 0
    for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):

        # print(type(labels[0]))
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(X_train)
        # outputs = outputs.to(torch.float32)
        # print(outputs[:][0])
        loss = criterion(outputs, y_train) 
        losses.append(loss)
        # print(loss)
        loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
        optimizer.step() # Updates weights and biases with the optimizer (SGD)

    return losses

if __name__ == "__main__": 
    # Set parameters here 
    input_shape = 10
    num_units = 128

    # rnn = build_model(input_shape, num_units)
    # train rnn 

    # Testing is 25% and Validation is 20%
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy( y_train).float()

    # rnn_model =  build_model(X_train.shape[2], 64)

    tiktok_model = TikTokModel(X_train.shape[2])

    # Set epochs and learning rate 
    learning_rate, epochs = set_hyperparameters()

    # Set optimizer and loss 
    criterion, optimizer = set_loss_and_optimizer(tiktok_model, learning_rate)

    # Train the model 
    # losses = train_model(optimizer, criterion, X_train, y_train, tiktok_model, epochs)
    # print(losses)

    losses = []
    # Iterations = []
    # Accuracy = []
    # iter = 0
    for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):

        # print(type(labels[0]))
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = tiktok_model(X_train)
        # outputs = outputs.to(torch.float32)
        # print(outputs[:][0])
        loss = criterion(outputs, y_train) 
        losses.append(loss)
        # print(loss)
        loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
        optimizer.step() # Updates weights and biases with the optimizer (SGD)

    torch.save(tiktok_model.state_dict(), 'trained_tiktok_model.pt')
    # print(vars(tiktok_model))
    print(losses)
    losses_converted = [y.detach().numpy() for y in losses]
    print(losses_converted)
    # losses = losses.detach().numpy()
    plt.plot(losses_converted)
    plt.show()