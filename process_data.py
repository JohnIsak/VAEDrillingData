import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
import pickle
import sklearn.preprocessing as pp

class DataSet(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _x = self.X[index]
        return _x

def make_sequential(x :pd.DataFrame, sequence_length):
    x = torch.tensor(x.to_numpy())
    x_seq = torch.zeros((x.shape[0] - sequence_length, sequence_length, x.shape[1]))
    for i in range(x.shape[0] - sequence_length):
        x_seq[i] = x[i:i + sequence_length]

    return x_seq


# DATA LAYOUT AS TENSOR BEFORE PROCESSING
# INDEX:        0       1       2       3       4
# Definition:   Number  mDepth  WOB     press   torque
# 5       6       7       8       9       10       11       12
# ROP     RPM     MudFlow MudDens Diamm   HookL   TVD     RAND
# When Dropping 0 the size should be 12

def process_data(sequence_length):
    names = ["data/USROP_A 0 N-NA_F-9_Ad.csv", "data/USROP_A 1 N-S_F-7d.csv", "data/USROP_A 2 N-SH_F-14d.csv",
             "data/USROP_A 3 N-SH-F-15d.csv", "data/USROP_A 3 N-SH-F-15d.csv", "data/USROP_A 5 N-SH-F-5d.csv"]
    train = None
    for name in names:
        data = pd.read_csv(name)
        data = data.drop(columns="Unnamed: 0")
        seq_data = make_sequential(data, sequence_length)
        if train == None:
            train = seq_data
        elif name == names[-1]:
            val = seq_data
        else:
            train = torch.cat([seq_data, train])


    scaler = pp.RobustScaler()

    train = train.reshape(-1, train.shape[-1])
    train = scaler.fit_transform(train)
    train = train.reshape(-1, sequence_length, train.shape[-1])
    val = val.reshape(-1, val.shape[-1])
    val = scaler.transform(val)
    val = val.reshape(-1, sequence_length, val.shape[-1])

    train = torch.tensor(train, dtype=torch.float32)
    val = torch.tensor(val, dtype=torch.float32)

    train = DataSet(train)
    val = DataSet(val)
    train = DataLoader(train, batch_size=128, shuffle=True)
    val = DataLoader(val, batch_size=128, shuffle=False)
    pickle.dump(scaler, open("models/data_transform.pkl", 'wb'))
    torch.save(train, f"data/train_seqlen_{sequence_length}")
    torch.save(val, f"data/val_seqlen_{sequence_length}")
    return train, val