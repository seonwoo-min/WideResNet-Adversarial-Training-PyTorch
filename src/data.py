# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - TorchVision

import os

import torchvision
import torchvision.transforms as transforms


dataset_info = {
    "cifar10":{
        "size": 32,
        "num_channels": 3,
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2023, 0.1994, 0.2010),
        "num_classes": 10
    },

    "cifar100": {
        "size": 32,
        "num_channels": 3,
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
        "num_classes": 100
    },
}


def get_dataset(dataset, test, sanity_check=False):
    info = dataset_info[dataset]

    if not test:
        transform = transforms.Compose([
            transforms.RandomCrop(info["size"], padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ])
        if dataset == "cifar10":
            download = not os.path.exists('./data/cifar-10-batches-py')
            dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
        elif dataset == "cifar100":
            download = not os.path.exists('./data/cifar-100-python')
            dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=download, transform=transform)

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ])
        if dataset == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        elif dataset == "cifar100":
            dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)

    if sanity_check:
        dataset.data = dataset.data[:320]

    return dataset, info