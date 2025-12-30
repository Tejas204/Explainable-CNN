import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, input_dims, activation, hidden_dims):
        super(SAE, self).__init__()
        self.activation = activation
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims

    def build_sae(self):
        self.encoder = nn.Linear(self.input_dims, self.hidden_dims)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(self.hidden_dims, self.input_dims)

    def forward(self, x):
        x = self.relu(self.encoder(x))
        x = self.decoder(x)
        return x
