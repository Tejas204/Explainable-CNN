# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


# Define transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load Dataset: CIFAR
train_dataset = torchvision.datasets.CIFAR10(root="/Users/tejasdhopavkar/Documents/DL/Explainable_CNN/data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="/Users/tejasdhopavkar/Documents/DL/Explainable_CNN/data", train=False, transform=transform, download=True)

class LoadData(Dataset):
    def __init__(self, data, labels, transform=transform):
        super(LoadData, self).__init__()
        self.transform = transform
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
