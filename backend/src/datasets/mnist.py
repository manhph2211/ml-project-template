from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader

data_path = 'C:/Usrs/phudn4/Downloads/normalizing-flow-with-jax/backend/src/data'

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))

mnist_val = datasets.MNIST(data_path, train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))

train_loader = DataLoader(mnist_train, batch_size=64,
                          shuffle=True, pin_memory=True, drop_last=True)

val_loader = DataLoader(mnist_val, batch_size=1,
                        shuffle=False, pin_memory=True, drop_last=True)