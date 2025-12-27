import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplainableCNN(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, activation, normalization, max_pool, drop_prob = 0.0):
        super(ExplainableCNN, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm = normalization
        self.drop_prob = drop_prob
        self.max_pool = max_pool

        self.build_model()


    def build_model(self):
        layers = []

        input_dims = self.input_size
        for i in range(len(self.hidden_layers)):
            # Convolution
            layers.append(nn.Conv2d(input_dims, self.hidden_layers[i], 3, stride=1, padding=1))

            # Batch Norm
            if self.norm:
                layers.append(self.norm(self.hidden_layers[i]))

            # Max pooling
            if self.max_pool and i == len(self.hidden_layers[i]) - 2:
                layers.append(nn.MaxPool2d((2, 2), stride=2))
            
            # Activation
            layers.append(self.activation())

            # Drop probabilities
            if self.drop_prob:
                layers.append(nn.Dropout(self.drop_prob))

            input_dims = self.hidden_layers[i]

        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_dims*32*32, self.num_classes))
        self.layers = nn.Sequential(*layers)

    
    def forward(self, x):
        output = self.layers(x)
        return output