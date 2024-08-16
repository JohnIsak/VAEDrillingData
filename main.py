import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from captum.attr import FeaturePermutation, FeatureAblation, IntegratedGradients
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import vae
import plots

class DataSet(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _x = self.X[index]
        return _x

def main():
    # make_sequential(30)
    # return 0

    train = torch.load("data/train_seqlen_30")
    val = torch.load("data/val_seqlen_30")
    model = vae.VAE()
    training(model, 30, train, val, 100)


def loss_function(recon_x, x, mu, log_var):
    # print(recon_x.shape)
    # print(x.shape)
    MSE = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    beta = 0.1
    KLD = beta * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))
    if np.random.rand() < 0.01:
        print("MSE: ", MSE, "KLD: ", KLD)
        print("mu: ", mu, "log_var: ", log_var)
        print("recon_x - x: ", recon_x - x)

    return MSE + KLD

def training(model, num_intervals, train, val, num_epochs):
    lr = 0.001
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_f = loss_function
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        print("epoch", epoch + 1)
        model.train()

        def forward(data, backprop=True):
            running_loss = 0.0
            for batch_idx, batch in enumerate(data):
                model.zero_grad()
                x = batch
                x_recon, mu, logvar = model(x)
                loss = loss_f(x_recon, x, mu, logvar)
                if backprop:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            running_loss = running_loss / batch_idx
            return running_loss

        running_loss = forward(train)

        train_loss[epoch] = running_loss
        print("Training loss: ", running_loss)
        model.eval()

        running_loss = forward(val, False)

        val_loss[epoch] = running_loss
        print("Validation loss: ", running_loss)
        if running_loss < best_val_loss:
            print("Saving model")
            torch.save(model, "models/" + model.__class__.__name__ + "_nInt" + str(num_intervals))
            best_val_loss = running_loss
    plots.plot_training(train_loss, val_loss, model)


def make_sequential(sequence_length):
    x = torch.load("data/X")
    y = torch.load("data/Y")

    print(y.shape)
    y = y.unsqueeze(1)
    x = torch.concat((x, y), 1)
    print(x)
    print(x.shape)

    x_seq = torch.zeros((x.shape[0] - sequence_length, sequence_length, x.shape[1]))

    for i in range(x.shape[0] - sequence_length):
        x_seq[i] = x[i:i + sequence_length]

    print(x_seq.shape)
    print(x_seq[0], "x_seq 0 ")

    train = DataSet(x_seq[:int(len(x)*0.8)])
    val = DataSet(x_seq[int(len(x)*0.8):])

    print(train.X[0], "train 0")
    print(train.X[1], "train 1")
    print(train.X[0].shape, "train 0 shape")
    print(train.X[1].shape, "train 1 shape")
    train = DataLoader(train, batch_size=128, shuffle=True)
    val = DataLoader(val, batch_size=128, shuffle=False)

    torch.save(train, "data/train_seqlen_" + str(sequence_length))
    torch.save(val, "data/val_seqlen_" + str(sequence_length))

    # Sanity check
    # for i in range(100):
    #    for j in range(30):
    #        print(x[i+j] - x_seq[i, j])

if __name__ == '__main__':
    main()

