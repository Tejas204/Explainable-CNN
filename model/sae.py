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
        """
        Initialize the CLIPSAE model for joint feature and CLIP embedding encoding.

        Args:
            device: The device to run the model on (e.g., 'cpu' or 'cuda').
            feature_input_dims (int): Dimensionality of the input features.
            activation: The activation function class to use (e.g., nn.ReLU).
            feature_hidden_dims (int): Dimensionality of the hidden layer for features.
            clip_embeddings (Tensor): Precomputed CLIP embeddings for the dataset.
        """
        super(CLIPSAE, self).__init__()
        self.activation = activation
        self.hidden_dims = feature_hidden_dims
        self.input_dims = feature_input_dims
        self.device = device
        self.clip_embeddings = clip_embeddings

    def build_clipsae(self):
        """
        Build the encoder layers for CLIP embeddings and features, and the decoder layer.
        """
        self.clip_encoder = nn.Linear(self.clip_embeddings.shape[0], self.hidden_dims)
        self.feature_encoder = nn.Linear(self.input_dims, self.hidden_dims)
        self.decoder = nn.Linear(self.hidden_dims, self.input_dims)
        self.relu = self.activation()

    def forward(self, feature, clip):
        """
        Forward pass for CLIPSAE.

        Args:
            feature: Input feature tensor.
            clip: Input CLIP embedding tensor.

        Returns:
            Reconstructed feature tensor.
        """
        feature = self.relu(self.feature_encoder(feature))
        clip = self.relu(self.clip_encoder(clip))
        feature = self.decoder(feature)
        return feature
    
class CLIPENCODER(nn.Module):
    def __init__(self, device, activation, input_dims, hidden_dims ,clip_embeddings):
        """
        Initialize the CLIPENCODER model for encoding CLIP embeddings.

        Args:
            device: The device to run the model on (e.g., 'cpu' or 'cuda').
            activation: The activation function class to use (e.g., nn.ReLU).
            input_dims (int): Dimensionality of the input CLIP embeddings.
            hidden_dims (int): Dimensionality of the hidden layer.
            clip_embeddings (Tensor): Precomputed CLIP embeddings for the dataset.
        """
        super(CLIPENCODER, self).__init__()
        self.device = device
        self.activation = activation
        self.clip_embeddings = clip_embeddings
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims

    def build_clip_encoder(self):
        """
        Build the encoder layer for CLIP embeddings.
        """
        self.clip_encoder = nn.Linear(self.input_dims, self.hidden_dims, device=self.device)
        self.relu = self.activation()

    def forward(self, x):
        """
        Forward pass for CLIPENCODER.

        Args:
            x: Input CLIP embedding tensor.

        Returns:
            Encoded CLIP embedding tensor.
        """
        clip_encoding = self.relu(self.clip_encoder(x))
        return clip_encoding