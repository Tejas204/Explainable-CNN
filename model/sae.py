import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, input_dims, activation, hidden_dims):
        """
        Initialize the Sparse Autoencoder (SAE) model.

        Args:
            input_dims (int): The dimensionality of the input features.
            activation: The activation function class to use (e.g., nn.ReLU).
            hidden_dims (int): The dimensionality of the hidden layer.
        """
        super(SAE, self).__init__()
        self.activation = activation
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims
        self.build_sae()

    def build_sae(self):
        """
        Build the encoder and decoder layers of the autoencoder.
        """
        self.encoder = nn.Linear(self.input_dims, self.hidden_dims)
        self.relu = self.activation()
        self.decoder = nn.Linear(self.hidden_dims, self.input_dims)

    def forward(self, x):
        """
        Perform a forward pass through the autoencoder.

        Args:
            x: Input tensor to be encoded and reconstructed.

        Returns:
            Reconstructed output tensor.
        """
        x = self.relu(self.encoder(x))
        x = self.decoder(x)
        return x
