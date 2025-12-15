# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader




class LoadData(Dataset):
    def __init__(self, transform):
        """
        Docstring for __init__
        
        :param self
        :param data: contains the train or test data
        :param labels: contains the labels of corresponding data
        :param transform: convert to Tensor and normalize
        """
        super(LoadData, self).__init__()
        self.transform = transform
    
    def __len__(self):
        """
        Docstring for __len__
        
        :param self
        """
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        Docstring for __getitem__
        
        :param self: Description
        :param index: Description
        """
        sample = self.data[index]
        label = self.labels[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    
    def load_data(self, dataset:str):
        """
        Docstring for load_data
        
        :param self: Description
        :param dataset: Description
        :type dataset: str
        """
        if dataset == "CIFAR10":
            training_data = torchvision.datasets.CIFAR10(root="/Users/tejasdhopavkar/Documents/DL/Explainable_CNN/data/CIFAR10", train=True, transform=self.transform, download=True)
            testing_data = torchvision.datasets.CIFAR10(root="/Users/tejasdhopavkar/Documents/DL/Explainable_CNN/data/CIFAR10", train=False, transform=self.transform, download=True)
        elif dataset == "CIFAR100":
            training_data = torchvision.datasets.CIFAR100(root="/Users/tejasdhopavkar/Documents/DL/Explainable_CNN/data/CIFAR100", train=True, transform=self.transform, download=True)
            testing_data = torchvision.datasets.CIFAR100(root="/Users/tejasdhopavkar/Documents/DL/Explainable_CNN/data/CIFAR100", train=False, transform=self.transform, download=True)
        return training_data, testing_data



