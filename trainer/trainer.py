# Imports
import sys
import os
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from config.config import CNN_Config

class Trainer():
    def __init__(self, criterion, optimizer, batch_size, epochs, train_loader, model, experiment):
        """Initialize the Trainer class.

        Args:
            criterion: The loss function used for training.
            optimizer: The optimizer for updating model parameters.
            batch_size: The size of each training batch.
            epochs: The number of training epochs.
            train_loader: The data loader for training data.
            model: The neural network model to be trained.
            experiment: The name or identifier of the experiment.
        """
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_loader = train_loader
        self.n_total_steps = len(self.train_loader)
        self.experiment = experiment
        print(f"Starting Experiment: {self.experiment}")

    def train_model(self):
        """Train the CNN model using the provided data loader.

        This method performs forward pass, computes loss, and updates model parameters
        for the specified number of epochs. Prints progress every 100 steps.
        """
        for epoch in range(self.epochs):
            for batch, (images, labels) in enumerate(self.train_loader):
                # Forward pass
                outputs = self.model(images)

                # Loss
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                if (batch + 1) % 100 == 0:
                    print(f"Epoch: {epoch+1} / {self.epochs}, step {batch+1}/{self.n_total_steps}, loss = {loss.item():.4f}")

        print("\nFinished Training!")


    def train_sae_model(self):
        """Train the SAE (Sparse Autoencoder) model using the created loaders.

        This method performs forward pass, computes reconstruction loss, and updates
        model parameters for the specified number of epochs. Prints progress every 100 steps.
        """
        for epoch in range(self.epochs):
            for batch, x in enumerate(self.train_loader):
                # Forward pass
                output = self.model(x)

                # Loss
                loss = self.criterion(output, x)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (batch + 1) % 100 == 0:
                    print(f"Epoch: {epoch+1} / {self.epochs}, step {batch+1}/{self.n_total_steps}, loss = {loss.item():.4f}")

        print(f"\nFinished training!")
