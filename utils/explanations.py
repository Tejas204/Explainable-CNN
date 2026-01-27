import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2gray

class Explanations():
    def __init__(self, activations):
        self.activations = activations

    def average(self, dim:int):
        self.average_activations = torch.mean(self.activations, dim=dim)

    def arrange(self, h:int, w:int):
        self.arranged_tensor = torch.reshape(self.average_activations, (h, w))

    def overlap_heatmap(self, image, activation, alpha_scale, cmap='jet', greyscale = False, single_feature=False):
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            if greyscale:
                image = rgb2gray(image)
                image = np.stack([image, image, image], axis=-1)

        if torch.is_tensor(activation):
            activation = activation.detach().cpu().numpy()

        # Normalize activation
        act = activation - activation.min()
        act = act / (act.max() + 1e-8)

        if not single_feature:
            colormap = plt.get_cmap(cmap)
            heatmap = colormap(act)[..., :3]

            alpha = (act * alpha_scale)[..., None]

            # Blend
            # overlay = (1 - alpha) * image + alpha * heatmap
            overlay = image*(1-alpha) + alpha*heatmap
            overlay = np.clip(overlay, 0, 1)
        else:
            rows = 32
            columns = 32
            
            fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
            axes = axes.flatten()

            for i in range(columns):
                axes[i].imshow(act[i].cpu(), cmap="viridis")
                axes[i].axis("off")
                axes[i].set_title(f"Ch {i}", fontsize=8)

        return overlay