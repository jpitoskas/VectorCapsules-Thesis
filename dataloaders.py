# import os, torch
# import torchvision
import numpy as np
# import scipy.io as sio

import torch
# from torchvision import datasets

from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
import os

from utils import Standardize, random_split, CustomDataset

def load_dataset(args):

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}


    if args.dataset == 'smallnorb':
        args.channels = 2
        args.n_classes = 5
        args.Hin = 32
        args.Win =32
        dataset = 'smallNORB_48'
        working_dir = os.path.join(os.getcwd(), 'data', dataset)

        dataset_paths = {'train': os.path.join(working_dir,'train'),
                        'test':  os.path.join(working_dir,'test')}

        dataloaders = smallnorb(args, dataset_paths)
        train_loader = dataloaders['train']
        test_loader = dataloaders['test']


        # args.class_names = ('car', 'animal', 'truck', 'airplane', 'human') # 0,1,2,3,4 labels
        # args.n_channels, args.n_classes = 2, 5

    elif args.dataset == 'mnist':
        args.channels = 1
        args.n_classes = 10
        args.Hin = 28
        args.Win =28
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(2), transforms.RandomCrop(28),
                               transforms.ToTensor()
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)


    return train_loader, test_loader





def smallnorb(args, dataset_paths):

    transf = {'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((args.crop_dim, args.crop_dim)),
                transforms.ColorJitter(brightness=args.brightness/255., contrast=args.contrast),
                transforms.ToTensor(),
                Standardize()]),
                # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))]),
        'test':  transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((args.crop_dim, args.crop_dim)),
                transforms.ToTensor(),
                Standardize()])}
                # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])}

    config = {'train': True, 'test': False}
    datasets = {i: smallNORB(dataset_paths[i], transform=transf[i],
        shuffle=config[i]) for i in config.keys()}

    # return data, labels dicts for new train set and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].labels,
        n_classes=5,
        n_samples_per_class=np.unique(
            datasets['train'].labels, return_counts=True)[1] // 5) # % of train set per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((args.crop_dim, args.crop_dim)),
            transforms.ColorJitter(brightness=args.brightness/255., contrast=args.contrast),
            transforms.ToTensor(),
            Standardize()])
            # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            Standardize()])
            # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    config = {'train': True, 'train_valid': True,
        'valid': False, 'test': False}

    dataloaders = {i: DataLoader(datasets[i], shuffle=config[i], pin_memory=True,
        num_workers=args.num_workers, batch_size=args.batch_size) for i in config.keys()}

    return dataloaders

class smallNORB(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label'''

    def __init__(self, data_path, shuffle=True, transform=None):

        self.data_path = data_path
        self.shuffle = shuffle
        self.transform = transform
        self.data, self.labels = [], []

        # get path for each class folder
        for class_label_idx, class_name in enumerate(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_name)

            # get name of each file per class and respective class name/label index
            for _, file_name in enumerate(os.listdir(class_path)):
                img = np.load(os.path.join(data_path, class_name, file_name))
                # Out ← [H, W, C] ← [C, H, W]
                if img.shape[0] < img.shape[1]:
                    img = np.moveaxis(img, 0, -1)
                self.data.extend([img])
                self.labels.append(class_label_idx)

        self.data = np.array(self.data, dtype=np.uint8)
        self.labels = np.array(self.labels, dtype=np.int64)

        if self.shuffle:
            # shuffle the dataset
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.data[idx])

        return image, self.labels[idx] # (X, Y)
