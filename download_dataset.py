from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True, parents=True)
    CIFAR10(root='data', download=True)
    CIFAR100(root='data', download=True)