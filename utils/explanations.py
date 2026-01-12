import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Explanations():
    def __init__(self, activations):
        self.activations = activations

    def average(self, dim:int):
        self.average_activations = torch.mean(self.activations, dim=dim)

    def arrange(self, h:int, w:int):
        self.arranged_tensor = torch.reshape(self.average_activations, (h, w))