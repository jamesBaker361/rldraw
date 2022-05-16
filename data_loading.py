
from torchvision import datasets, transforms
import torch
from string_globals import *
import numpy as np

def make_rgb(img):
    return img *255

def get_data_loaders(mnist_dir,image_size,batch_size,channels=1,rgb=False):
    trans_list=[]
    if rgb:
        trans_list+=make_rgb
    trans_list+=[transforms.Resize(image_size),more_channels(channels),transforms.ToTensor()
        #,transforms.Normalize(mean=(0.5), std=(0.5))
    ]



    # MNIST Dataset
    transform = transforms.Compose(trans_list)

    train_dataset = datasets.MNIST(root=mnist_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=mnist_dir, train=False, transform=transform, download=True)
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def more_channels(channels):
    def _more_channels(arr):
        return np.stack((arr,)*channels, axis=-1)
    return _more_channels

def get_data_loaders_specific_char(mnist_dir,image_size,batch_size,char,channels=1,rgb=False):

    if char<0 or char>9:
        return get_data_loaders(mnist_dir,image_size,batch_size,channels)

    trans_list=[]
    if rgb:
        trans_list+=make_rgb
    trans_list+=[transforms.Resize(image_size),more_channels(channels),transforms.ToTensor()
        #,transforms.Normalize(mean=(0.5), std=(0.5))
    ]



    # MNIST Dataset
    transform = transforms.Compose(trans_list)

    train_dataset = datasets.MNIST(root=mnist_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=mnist_dir, train=False, transform=transform, download=True)

    for dataset in [train_dataset,test_dataset]:
        indices = dataset.targets == char # if you want to keep images with the label char
        dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    for c in range(11):
        train,test=get_data_loaders_specific_char(mnist_dir,64,4,c,3)
        print(c,len(train.dataset),len(test.dataset))