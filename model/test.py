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

if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2
    )

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    trained_tiktok_model = TikTokModel(X_test.shape[2])
    trained_tiktok_model.load_state_dict(torch.load("trained_tiktok_model.pt"))
    print(trained_tiktok_model)

    outputs_test = trained_tiktok_model(X_test)
    outputs_test = torch.squeeze(outputs_test)

    test_data_labels = y_test
    test_data_labels = test_data_labels.to(torch.float32)

    predicted_test = outputs_test.round().detach().numpy()
    total_test = test_data_labels.size(0)

    correct_test = np.sum(predicted_test == test_data_labels.detach().numpy())
    accuracy_test = 100 * correct_test / total_test

    print(accuracy_test)
