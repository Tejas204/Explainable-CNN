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

    def overlap_heatmap(self, image, activation, alpha, cmap='jet'):
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
        if torch.is_tensor(activation):
            activation = activation.detach().cpu().numpy()

        # Normalize activation
        act = activation - activation.min()
        act = act / (act.max() + 1e-8)

        colormap = plt.get_cmap(cmap)
        heatmap = colormap(act)[..., :3]

        # if image.max() > 1:
        #     image = image / 255.0

        # Blend
        # overlay = (1 - alpha) * image + alpha * heatmap
        overlay = image + alpha*heatmap
        # overlay = np.clip(overlay, 0, 1)

        return overlay