# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define hyperparameters and layers
batch_size = 20

# Experiment 1
CNN_Config = {
    "input_size": 28,
    "output_size": 3,
    "num_classes": 10,
    "activation": nn.ReLU,
    "normalization": True,
    "drop_prob": 0.4,
    "hidden_dims": [64, 128, 256, 128, 64]
}