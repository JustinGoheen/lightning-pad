import os
from torchvision.datasets import MNIST
from torchvision import transforms


if __name__ == "__main__":
    rootdir = os.getcwd()
    datapath = os.path.join(rootdir, "data", "cache")
    MNIST(datapath, train=True, download=True, transform=transforms.ToTensor())
    MNIST(datapath, train=False, download=True, transform=transforms.ToTensor())