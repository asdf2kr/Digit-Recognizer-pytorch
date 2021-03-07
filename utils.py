import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MNIST(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels.values
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = np.array(self.images.iloc[idx, :]).astype(np.uint8).reshape(-1, 28, 28)

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img

def prepare_dataloaders(args):
    data_path = os.path.join(os.getcwd(), 'Datas')

    train_csv = pd.read_csv(os.path.join(data_path, 'train.csv'))

    # Split the train set so there is also a validation set
    train_images, val_images, train_labels, val_labels = train_test_split(train_csv.iloc[:,1:],
                                                                            train_csv.iloc[:,0],
                                                                            test_size = 0.1)
    train_transform = torchvision.transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], # 1 for grayscale channels
                                                    std=[0.5])
                           ])

    valid_transform = torchvision.transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], # 1 for grayscale channels
                                                    std=[0.5])
                           ])

    train_dataset = MNIST(images = train_images,
                        labels = train_labels,
                        transform=train_transform)

    valid_dataset = MNIST(images = val_images,
                        labels = val_labels,
                        transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size = args.batch_size,
                                                num_workers = args.workers,
                                                shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size = args.batch_size,
                                                num_workers = args.workers)


    return train_loader, valid_loader, len(train_dataset), len(valid_dataset)


def prepare_test_dataloaders(args):
    data_path = os.path.join(os.getcwd(), 'Datas')
    test_csv = pd.read_csv(os.path.join(data_path, 'test.csv'))

    test_transform = torchvision.transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], # 1 for grayscale channels
                                                    std=[0.5])
                           ])

    test_dataset = MNIST(images = test_csv,
                        labels = None,
                        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size = 1,
                                                num_workers = args.workers)

    return test_loader, len(test_dataset)
