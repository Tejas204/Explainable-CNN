# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hooks():
    def __init__(self, model:nn.Module, module_names):
        """
        Initialize the Hooks class to register forward hooks on specified modules.

        Args:
            model (nn.Module): The PyTorch model to hook into.
            module_names (list): List of module names to register hooks for.
        """
        self.model = model
        self.module_names = module_names
        self.activations = {}
        self.hooks = []

        for name, module in model.named_modules():
            if name in self.module_names:
                self.hooks.append(
                    module.register_forward_hook(self.get_activations(name=name))
                )
    
    def get_activations(self, name:str):
        """
        Create a forward hook function to capture activations for a given module.

        Args:
            name (str): The name of the module.

        Returns:
            function: A hook function that stores the output in self.activations.
        """
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
        
    def remove(self):
        """
        Remove all registered hooks from the model.
        """
        for hook in self.hooks:
            hook.remove()