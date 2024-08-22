import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from captum.attr import FeaturePermutation, FeatureAblation, IntegratedGradients
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
import seaborn as sb
import matplotlib.pyplot as plt
import process_data
import numpy as np
import vae
import plots

reduced_indices = True
big_loss_indexes = [5611, 7024, 7763] # Loss indexes when including RPM and WOB
# loss_indices_non_WOB_RPM = [2016, 2058, 2142, 2145, 2147, 2152, 2154, 3825, 4162, 4190, 5323, 5709, 7018]
loss_indices_non_control = [8819, 15230]

indices_non_control_parameters = torch.tensor([0, 2, 3, 4, 8, 9, 10, 11])



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
    # process_data.process_data(30)
    # return 0


    # val = torch.load("data/val_seqlen_30")
    # forward_val(val, torch.load("models/VAE_indecestensor([ 0,  2,  3,  4,  8,  9, 10, 11])"))
    # return 0

    # val = torch.load("data/val_seqlen_30")
    # plots.plot_anomaly(val, loss_indices_non_control)
    # return 0

    val = torch.load("data/val_seqlen_30")
    model = torch.load("models/VAE_indecestensor([ 0,  2,  3,  4,  8,  9, 10, 11])")
    plots.plot(val, model, loss_indices_non_control)
    return 0

    train = torch.load("data/train_seqlen_30")
    val = torch.load("data/val_seqlen_30")
    model = vae.VAE()
    training(model, 30, train, val, 100)

def forward_val(data :DataLoader, model: nn.Module):
    saved_index = 0
    max_loss = 0
    indexes = []
    for i in range(data.dataset.X.shape[0]):
        x = data.dataset.X[i].unsqueeze(0)
        x_recon, mu, logvar = model.forward(x)

        if reduced_indices:
            x_recon = torch.index_select(x_recon, 2, indices_non_control_parameters)
            x = torch.index_select(x, 2, indices_non_control_parameters)

        loss = F.mse_loss(x_recon, x)
        if loss > 0.75:

            print(saved_index, "saved index")
            print(i, "current index")
            if saved_index + 1 == i:
                saved_index = i
                print(loss, "loss")
                print(max_loss, "max loss")
                if loss > max_loss:
                    max_loss = loss
                    indexes[-1] = i
                    print(indexes, "indexes")
            else:
                print("appending")
                indexes.append(i)
                max_loss = loss
                saved_index = i
    print(indexes)

def loss_function(recon_x, x, mu, log_var):
    # print(recon_x.shape)
    # print(x.shape)

    # When utilizing
    if reduced_indices:
        recon_x = torch.index_select(recon_x, 2, indices_non_control_parameters)
        x = torch.index_select(x, 2, indices_non_control_parameters)

    MSE = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    beta = 0.1
    KLD = beta * (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()))

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
            torch.save(model, "models/" + model.__class__.__name__ + "_indeces" + str(indices_non_control_parameters))
            best_val_loss = running_loss
    plots.plot_training(train_loss, val_loss, model)

if __name__ == '__main__':
    main()

