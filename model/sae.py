import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, device, input_dims, activation, hidden_dims):
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
        self.device = device
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

class CLIPSAE(nn.Module):
    def __init__(self, device, feature_input_dims, activation, feature_hidden_dims, clip_embeddings):
        super(CLIPSAE, self).__init__()
        self.activation = activation
        self.hidden_dims = feature_hidden_dims
        self.input_dims = feature_input_dims
        self.device = device
        self.clip_embeddings = clip_embeddings

    def build_clipsae(self):
        self.clip_encoder = nn.Linear(self.clip_embeddings.shape[0], self.hidden_dims)
        self.feature_encoder = nn.Linear(self.input_dims, self.hidden_dims)
        self.decoder = nn.Linear(self.hidden_dims, self.input_dims)
        self.relu = self.activation()

    def forward(self, feature, clip):
        feature = self.relu(self.feature_encoder(feature))
        clip = self.relu(self.clip_encoder(clip))
        feature = self.decoder(feature)
        return feature