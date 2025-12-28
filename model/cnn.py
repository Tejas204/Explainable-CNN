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
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64*32*32, 10)

        # layers = []

        # input_dims = self.input_size
        # for i in range(len(self.hidden_layers)):
        #     # Convolution
        #     layers.append(nn.Conv2d(input_dims, self.hidden_layers[i], 3, stride=1, padding=1))

        #     # Batch Norm
        #     if self.norm:
        #         layers.append(self.norm(self.hidden_layers[i]))

        #     # Max pooling
        #     if self.max_pool and i == len(self.hidden_layers[i]) - 2:
        #         layers.append(nn.MaxPool2d((2, 2), stride=2))
            
        #     # Activation
        #     layers.append(self.activation())

        #     # Drop probabilities
        #     if self.drop_prob:
        #         layers.append(nn.Dropout(self.drop_prob))

        #     input_dims = self.hidden_layers[i]

        # layers.append(nn.Flatten())
        # layers.append(nn.Linear(input_dims*32*32, self.num_classes))
        # self.layers = nn.Sequential(*layers)
        # nn.Sequential()

    
    def forward(self, x):
        # output = self.layers(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = F.relu(self.norm5(self.conv5(x)))
        x = self.flatten(x)
        x = self.linear(x)
        return x