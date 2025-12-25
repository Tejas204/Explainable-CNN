# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define hyperparameters and layers
batch_size = 20

# Experiment 1
CNN_Config = dict(
    name = "CNN_Config_1",

    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [64, 128, 256, 128, 64],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.4,
        max_pool = False
    )
)