import torch

from torchvision.datasets import CIFAR10, CIFAR100

import albumentations as A
import numpy as np


def convert_to_dataloader(datasets, batch_size, num_workers, train=True):
    return [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                                        drop_last=True) for ds in datasets]


def get_dataset(dataset_name):
    train_ds = get_my_dataset(dataset_name, train=True)
    valid_ds = get_my_dataset(dataset_name, train=False)
    test_ds = get_my_dataset(dataset_name, train=False)

    print('{} dataset class num: {}'.format(dataset_name, len(train_ds.classes)))
    print('{} train dataset len: {}'.format(dataset_name, len(train_ds)))
    print('{} valid dataset len: {}'.format(dataset_name, len(valid_ds)))
    print('{} test dataset len: {}'.format(dataset_name, len(test_ds)))

    return train_ds, valid_ds, test_ds


def get_my_dataset(dataset_name, train):
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root='data', train=train, download=False)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(root='data', train=train, download=False)
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

    transforms_train_valid = A.Compose([
        A.RandomSizedCrop([30, 32], 32, 32),
        A.HorizontalFlip(),
        A.Normalize(mean=mean, std=std)
    ])

    my_dataset = MyDataset(ds=dataset, transforms=transforms_train_valid)

    return my_dataset


class MyDataset(object):
    def __init__(self, ds, transforms=None):
        super().__init__()
        self.ds = ds
        self.transforms = transforms
        self.classes = ds.classes
        self.class_to_idx = ds.class_to_idx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]

        if self.transforms:
            img = np.transpose(self.transforms(image=np.array(img))['image'], (2, 0, 1))

        return img, label
