# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define hyperparameters and layers

# ------------------------------------------------------------------------------------
# Experiment 1 - Explainable CNN
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

# ------------------------------------------------------------------------------------
# Experiment 2 - SAE
# ------------------------------------------------------------------------------------
SAE_Config = dict(
    name = "SAE_Experiment_1",

    model_args = dict(
        input_dims = 64,
        hidden_dims = 125,
        epochs = 40,
        batch_size = 1,
        learning_rate = 0.001
    )
)

# ------------------------------------------------------------------------------------
# Experiment 3 - CLIP-SAE
# ------------------------------------------------------------------------------------
CLIPSAE_Configdict = dict(
    name = "SAE_Experiment_3",

    model_args = dict(
        input_dims = 64,
        hidden_dims = 100,
        epochs = 40,
        batch_size = 1,
        learning_rate = 0.001
    )
)