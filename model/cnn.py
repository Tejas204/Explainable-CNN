import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplainableCNN(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, activation, normalization, drop_prob = 0.0):
        super(ExplainableCNN, self).__init__()
        self.input_size = input_size
        self.output_size = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm = normalization
        self.drop_prob = drop_prob


    def build_model():
        pass

