# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hooks():
    def __init__(self):
        pass
    
    def get_activations(self, name:str):
        self.activations = {}
        def hook(model, input, output):
            self.activations[name] = output.detach()
            return self.activations
        return hook
        