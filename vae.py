import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.gru_encode = nn.GRU(12, 512, 1, batch_first=True)
        self.gru_decode = nn.GRU(12, 512, 1, batch_first=True)

        # VAE Encode
        self.fc1 = nn.Linear(512, 256)
        self.fc11 = nn.Linear(256, 64)
        self.fc12 = nn.Linear(256, 64)

        # VAE Decode
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 512)

        # Transform
        self.fc4 = nn.Linear(512, 12)

        self.name = "NN Model"

    def forward(self, x):
        seq_len = x.shape[1]
        _, hidden = self.gru_encode(x)
        x = F.leaky_relu(hidden[0])
        x = self.fc1(x)
        x = F.leaky_relu(x)
        mu = self.fc11(x)
        log_var = self.fc12(x)
        std = torch.exp(0.5*log_var)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        x = self.fc2(z)
        x = F.leaky_relu(x)
        hidden = self.fc3(x)
        hidden = hidden.view(1, hidden.shape[0], 512)
        seq = torch.zeros((x.shape[0], seq_len, 12), device=x.device)
        y = torch.zeros((x.shape[0], 1, 12), device=x.device)
        for i in range(seq_len):
            y, hidden = self.gru_decode(y, hidden)
            y = self.fc4(y)
            seq[:, i, :] = y.squeeze()
        return seq, mu, log_var