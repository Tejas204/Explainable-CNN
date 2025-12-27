# Imports
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import CNN_Config

class Trainer():
    def __init__(self, criterion, optimizer, batch_size, epochs, train_loader, model, experiment):
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
