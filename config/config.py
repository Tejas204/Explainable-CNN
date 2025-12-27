# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define hyperparameters and layers

# ------------------------------------------------------------------------------------
# Experiment 1
# ------------------------------------------------------------------------------------
CNN_Config = dict(
    name = "CNN_Experiment_1",

    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [64, 128, 256, 128, 64],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.4,
        max_pool = False
    ),

    criterion = nn.CrossEntropyLoss(),
    learning_rate = 0.01,
    batch_size = 10,
    epochs = 1
)