# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from config.config import batch_size
from dataloader.load_data import LoadData


# Define transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load Dataset
data = LoadData(transform=transform)
training_data, testing_data = data.load_data("CIFAR10")

# Data Loaders
train_loader = data.data_loaders(training_data, "train")
test_loader = data.data_loaders(testing_data, "test")

# Example visualization
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

