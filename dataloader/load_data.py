# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from config.config import CNN_Config
import math



class LoadData(Dataset):
    """
    A class used to load data

    ----------------------------------------------------------------
    Attributes
    ----------------------------------------------------------------

    transform: object
        The transform you want to apply to your dataset

    ----------------------------------------------------------------
    Methods
    ----------------------------------------------------------------

    __len__
        Returns the length of the dataset

    __getitem__
        Fetches a specific item of the dataset at given index

    load_data
        Loads data available with torchvision
    """
    def __init__(self, transform):
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

    def data_loaders(self, dataset, type):
        if type == "train":
            shuffle = True
        else:
            shuffle = False

        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=CNN_Config['batch_size'], shuffle=shuffle)
        return loader
    
    def create_sae_loaders(self, features:list, batch_size:int):
        """Create training and testing data loaders for the SAE model.

        Args:
            features (list): The list of feature data.
            batch_size (int): The size of each batch.

        This method splits the features into train and test sets (80% train, 20% test),
        and creates batch dictionaries for each.
        """
        # Length of features
        data_length = len(features)

        # Create train and test features
        train_features = features[:math.floor(data_length*0.2)]
        test_features = features[math.floor(data_length*0.2):]
        num_train_features = len(train_features)
        num_test_features = len(test_features)

        # Compute number of batches
        num_train_batches = math.floor(num_train_features / batch_size)
        num_test_batches = math.floor(num_test_features / batch_size)

        for j in range(2):
            batch_dictionary = {}
            start = 0
            end = batch_size
            size = num_train_batches if j == 0 else num_test_batches

            for i in range(size):
                batch_dictionary[i] = features[start: end]
                start = end
                end += batch_size
            
            if j == 0:
                self.train_loader = batch_dictionary
            else:
                self.test_loader = batch_dictionary


