import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataLoaderHelper(object):
    def __init__(self, dataloader):
        self.loader = dataloader
        self.dataiter = iter(self.loader)

    def __next__(self):
        try:
            data, target = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            data, target = next(self.dataiter)
        
        return data, target

class RandomPartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in partition_sizes:
            part_len = round(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)

class LabelwisePartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        # sizes is a class_num * vm_num matrix
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()
        # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
        for class_idx in range(len(data.classes)):
            label_indexes.append(list(np.where(np.array(data.targets) == class_idx)[0]))
            class_len.append(len(label_indexes[class_idx]))
            rng.shuffle(label_indexes[class_idx])
        
        # distribute class indexes to each vm according to sizes matrix
        for class_idx in range(len(data.classes)):
            begin_idx = 0
            for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                end_idx = begin_idx + round(frac * class_len[class_idx])
                end_idx = int(end_idx)
                self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)

def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=True, num_workers=4):
    if selected_idxs == None:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    else:
        partition = Partition(dataset, selected_idxs)
        dataloader = DataLoader(partition, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    
    return DataLoaderHelper(dataloader)

def load_datasets(dataset_type, data_path="/data/wxlou/dataset"):
    
    train_transform = load_default_transform(dataset_type, train=True)
    test_transform = load_default_transform(dataset_type, train=False)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100(data_path, train = True,
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR100(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'MNIST':
        train_dataset = datasets.MNIST(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.MNIST(data_path, train = False, 
                                            download = True, transform=train_transform)
    elif dataset_type == 'EMNIST':
        train_dataset = datasets.ImageFolder('/data/ymliao/data/emnist/byclass_train', transform = train_transform)
        test_dataset = datasets.ImageFolder('/data/ymliao/data/emnist/byclass_test', transform = train_transform)
    
    elif dataset_type == 'tinyImageNet':
        train_dataset = datasets.ImageFolder('/data/ymliao/data/tiny-imagenet-200/train', transform = train_transform)
        test_dataset = datasets.ImageFolder('/data/ymliao/data/tiny-imagenet-200/val', transform = train_transform)

    return train_dataset, test_dataset

def load_default_transform(dataset_type, train=False):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        if train:
            dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'CIFAR100':
        # reference:https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            dataset_transform = transforms.Compose([
                                transforms.RandomCrop(32, 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),
                                normalize
                            ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'FashionMNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
    
    elif dataset_type == 'MNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])

    elif dataset_type == 'EMNIST':
        dataset_transform =  transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))  
                        ])
    
    elif dataset_type == 'tinyImageNet':
        dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    return dataset_transform

def load_customized_transform(dataset_type):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])

    elif dataset_type == 'CIFAR100':
          dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(1.0),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])

    elif dataset_type == 'FashionMNIST':
          dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(1.0),
                           transforms.ToTensor()
                         ])
    
    elif dataset_type == 'MNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
                       
    return dataset_transform
