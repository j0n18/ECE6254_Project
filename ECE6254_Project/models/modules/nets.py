import torch
from torch import nn

class MLP_Encoder(nn.Module):
    def __init__(self, 
    input_dim, 
    hidden_size,
    output_dim, 
    drop_prop):

        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.drop_prop = drop_prop

        self.dropout = nn.Dropout(p = self.drop_prop)
        self.linear1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, 2*output_dim, bias=True)

    def forward(self, x):

        x = self.dropout(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        mu, logvar = torch.split(x, self.output_dim, dim=1)
        return mu, logvar

class MLP_Decoder(nn.Module):
    def __init__(self, 
    input_dim, 
    hidden_size,
    output_dim, 
    drop_prop):
    
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.drop_prop = drop_prop

        self.dropout = nn.Dropout(p = self.drop_prop)
        self.linear1 = nn.Linear(input_dim, hidden_size, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, output_dim, bias=False)
        self.dropout = nn.Dropout(p = self.drop_prop)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x