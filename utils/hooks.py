# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hooks():
    def __init__(self, model:nn.Module, module_names, k:int=None):
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
        self.feature_vectors = []
        self.k = k


    def make_forward_hook(self):
        """
        Register forward hooks on the specified modules to capture their activations.

        This method iterates through the model's named modules and registers a forward hook
        for each module whose name is in self.module_names. The hook captures the output
        activations and stores them in self.activations.
        """
        for name, module in self.model.named_modules():
            if name in self.module_names:
                self.hooks.append(
                    module.register_forward_hook(self.get_activations(name))
                )
                

    def make_forward_pre_hook(self):
        """
        Register forward pre-hooks on the specified modules to capture and sparsify activations.

        This method iterates through the model's named modules and registers a forward pre-hook
        for each module whose name is in self.module_names. The hook captures the input
        activations, applies top-k sparsification, and stores them in self.activations.
        """
        for name, module in self.model.named_modules():
            if name in self.module_names:
                self.hooks.append(
                    module.register_forward_pre_hook(self.sae_get_top_k_activations(name=name, k=self.k))
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
    
    def sae_get_top_k_activations(self, name:str, k:int):
        """
        Create a forward hook function to capture and sparsify activations using top-k selection.

        This method creates a hook that modifies the output activations by keeping only the
        top-k values and setting the rest to zero, enforcing sparsity in the activations.

        Args:
            name (str): The name of the module.
            k (int): The number of top activations to keep.

        Returns:
            function: A hook function that stores the sparsified output in self.activations.
        """
        def hook(module, input):
            (z, ) = input

            values, indices = torch.topk(z, k, dim=-1)

            mask = torch.zeros_like(z)
            mask.scatter_(dim=-1, index=indices, value=1.0)

            z_sparse = z * mask

            self.activations[name] = z_sparse

            return (z_sparse, )

        return hook
        
    def remove(self):
        """
        Remove all registered hooks from the model.
        """
        for hook in self.hooks:
            hook.remove()


    def collect_features(self, name:str):
        """
        Collect feature vectors from the activations of a specified module.

        This method extracts feature vectors from the activation map of the given module
        by iterating over each spatial position (height and width) and collecting the
        activation values across all feature dimensions for the first batch item.

        Args:
            name (str): The name of the module whose activations to collect features from.
                        Note: Currently hardcoded to use 'conv5' activations.

        Note:
            This method appends feature vectors to self.feature_vectors list.
            It only processes the first item in the batch (index 0), because there
            is only one item per batch
        """
        batch = self.activations[name].shape[0]
        dims = self.activations[name].shape[1]
        height = self.activations[name].shape[2]
        width =self.activations[name].shape[3]

        for h in range(height):
            for w in range(width):
                feature_array = []
                for dim in range(dims):
                    feature_array.append(self.activations[name][0][dim][h][w])
                self.feature_vectors.append(torch.tensor(feature_array))

    def write_features(self, filepath:str):
        """
        Write the collected feature vectors to a file.

        This method appends each feature vector in self.feature_vectors to the specified file,
        with each vector written as a string representation followed by a newline.

        Args:
            filepath (str): The path to the file where feature vectors will be written.
                            The file is opened in append mode ('a+').
        """
        with open(filepath, "a+") as file:
            for i in range(len(self.feature_vectors)):
                file.write(str(self.feature_vectors[i])+"\n")
            file.close()