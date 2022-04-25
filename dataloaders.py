# import os, torch
# import torchvision
import numpy as np
# import scipy.io as sio

import torch
# from torchvision import datasets
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os

from utils import Standardize, CustomDataset
from utils import random_split, sample_weights

def load_dataset(args):

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}


    if args.dataset == 'smallnorb':
        args.channels = 2
        args.n_classes = 5
        args.Hin = args.crop_dim
        args.Win = args.crop_dim
        dataset = 'smallNORB_48'
        working_dir = os.path.join(os.getcwd(), 'data', dataset)

        dataset_paths = {'train': os.path.join(working_dir,'train'),
                        'test':  os.path.join(working_dir,'test')}

        dataloaders = smallnorb(args, dataset_paths)


        # args.class_names = ('car', 'animal', 'truck', 'airplane', 'human') # 0,1,2,3,4 labels
        # args.n_channels, args.n_classes = 2, 5

    elif args.dataset == 'mnist':
        args.channels = 1
        args.n_classes = 10
        args.Hin = args.crop_dim
        args.Win = args.crop_dim
        dataset = "MNIST"

        working_dir = os.path.join(os.getcwd(), 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                        'test':  os.path.join(working_dir,'test')}
        dataloaders = mnist(args, dataset_paths)

        # train_loader = torch.utils.data.DataLoader(
        #     torchvision.datasets.MNIST('../data', train=True, download=True,
        #                    transform=transforms.Compose([
        #                        transforms.Pad(2), transforms.RandomCrop(28),
        #                        transforms.ToTensor()
        #                    ])),
        #     batch_size=args.batch_size, shuffle=True, **kwargs)
        #
        # test_loader = torch.utils.data.DataLoader(
        #     torchvision.datasets.MNIST('../data', train=False, transform=transforms.Compose([
        #         transforms.ToTensor()
        #     ])),
        #     batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'svhn':
        args.channels = 3
        args.n_classes = 10
        dataset = 'SVHN'
        args.Hin = args.crop_dim
        args.Win = args.crop_dim
        working_dir = os.path.join(os.getcwd(), 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                        'test':  os.path.join(working_dir,'test')}
        dataloaders = svhn(args, dataset_paths)



    elif args.dataset == 'fashionmnist':
        args.channels = 1
        args.n_classes = 10
        dataset = 'FashionMNIST'
        args.padding = 4
        args.Hin = args.crop_dim
        args.Win = args.crop_dim
        working_dir = os.path.join(os.getcwd(), 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                        'test':  os.path.join(working_dir,'test')}
        dataloaders = fashionmnist(args, dataset_paths)

    elif args.dataset == 'cifar10':
        args.channels = 3
        args.n_classes = 10
        dataset = 'CIFAR10'
        args.padding = 4
        args.Hin = args.crop_dim
        args.Win = args.crop_dim
        working_dir = os.path.join(os.getcwd(), 'data', dataset)
        dataset_paths = {'train': os.path.join(working_dir,'train'),
                        'test':  os.path.join(working_dir,'test')}
        dataloaders = cifar10(args, dataset_paths)





    return dataloaders['train'], dataloaders['valid'], dataloaders['test']





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


def mnist(args, dataset_paths):
    ''' Loads the MNIST dataset.
        Returns: train/valid/test set split dataloaders.
    '''
    transf = {'train': transforms.Compose([
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                # transforms.Normalize((0.1307,), (0.3081,))
                ]),
        'test':  transforms.Compose([
                transforms.Pad(np.maximum(0, (args.crop_dim-28) // 2)),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                # transforms.Normalize((0.1307,), (0.3081,))
                ])}

    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.MNIST(root=dataset_paths[i], transform=transf[i],
        train=config[i], download=True) for i in config.keys()}

    # split train into train and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].targets,
        n_classes=10,
        n_samples_per_class=np.repeat(500, 10)) # 500 per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad(np.maximum(0, (args.crop_dim-28) // 2)),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ])

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


    dataloaders = {i: DataLoader(datasets[i], num_workers=args.num_workers, pin_memory=True,
        batch_size=args.batch_size, shuffle=config[i]) for i in config.keys()}

    if args.test_affNIST:
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', 'affNIST')

        aff_transf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))])

        datasets['affNIST_test'] = affNIST(data_path=os.path.join(working_dir,'test'),
            transform=aff_transf)
        dataloaders['affNIST_test'] = DataLoader(datasets['affNIST_test'], pin_memory=True,
            num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    return dataloaders



def svhn(args, dataset_paths):
    ''' Loads the SVHN dataset.
        Returns: train/valid/test set split dataloaders.
    '''
    transf = {
        'train': transforms.Compose([
            transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
            transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
            transforms.ToTensor(),
            # Standardize()]),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                 (0.19803012, 0.20101562, 0.19703614))]),
        # 'extra': transforms.Compose([
        #     transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
        #     transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
        #     transforms.ToTensor(),
        #     # Standardize()]),
        #     transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
        #                          (0.19803012, 0.20101562, 0.19703614)),
        'test': transforms.Compose([
            transforms.ToTensor(),
            # Standardize()])}
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                 (0.19803012, 0.20101562, 0.19703614))])
    }

    # config = {'train': True, 'extra': True, 'test': False}
    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.SVHN(root=dataset_paths[i], transform=transf[i],
                        split=i, download=True) for i in config.keys()}

    # weighted sampler weights for full(f) training set
    f_s_weights = sample_weights(datasets['train'].labels)


    # print(np.repeat(1000, 10).reshape(-1))
    #
    # exit()
    # return data, labels dicts for new train set and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
                                labels=datasets['train'].labels,
                                n_classes=10,
                                n_samples_per_class=np.repeat(1000, 10).reshape(-1))

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
        transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
        transforms.ToTensor(),
        # Standardize()])
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                             (0.19803012, 0.20101562, 0.19703614))])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # Standardize()])
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                             (0.19803012, 0.20101562, 0.19703614))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make channels last and convert to np arrays
    data['train'] = np.moveaxis(np.array(data['train']), 1, -1)
    data['valid'] = np.moveaxis(np.array(data['valid']), 1, -1)

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['valid'])

    # weighted sampler weights for new training set
    s_weights = sample_weights(datasets['train'].labels)

    config = {
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size) for i in config.keys()}

    #NOTE: comment these to use the weighted sampler dataloaders above instead
    config = {'train': True, 'train_valid': True,
        'valid': False, 'test': False}

    dataloaders = {i: DataLoader(datasets[i], num_workers=args.num_workers, pin_memory=True,
        batch_size=args.batch_size, shuffle=config[i]) for i in config.keys()}

    return dataloaders

def fashionmnist(args, dataset_paths):
    ''' Loads the Fashion-MNIST dataset.
        Returns: train/valid/test set split dataloaders.
    '''
    transf = {
        'train': transforms.Compose([
            transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
            # transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
            transforms.ToTensor(),
            # Standardize()]),
            transforms.Normalize((0.2860406,), (0.3530242,))
            ]),

        'test': transforms.Compose([
            transforms.Pad(padding=(args.crop_dim - 28) // 2),
            transforms.ToTensor(),
            # Standardize()])}
            transforms.Normalize((0.2860406,), (0.3530242,))
            ])
    }

    # config = {'train': True, 'extra': True, 'test': False}
    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.FashionMNIST(root=dataset_paths[i], transform=transf[i],
                        train=config[i], download=True) for i in config.keys()}

    # split train into train and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].targets,
        n_classes=10,
        n_samples_per_class=np.repeat(500, 10)) # 500 per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                transforms.Normalize((0.2860406,), (0.3530242,))
                ])


    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                transforms.Normalize((0.2860406,), (0.3530242,))
                ])

    # transf['train_'] = transf['train']
    # transf['valid'] = transf['train']



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


    dataloaders = {i: DataLoader(datasets[i], num_workers=args.num_workers, pin_memory=True,
        batch_size=args.batch_size, shuffle=config[i]) for i in config.keys()}


    return dataloaders


def cifar10(args, dataset_paths):
    ''' Loads the CIFAR-10 dataset.
        Returns: train/valid/test set split dataloaders.
    '''
    transf = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
            # transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # Standardize()]),
            transforms.Normalize((0.4913997, 0.4821584, 0.4465309),
                                 (0.2470322, 0.2434851, 0.2615878)),
            ]),

        'test': transforms.Compose([
            transforms.ToTensor(),
            # Standardize()])}
            transforms.Normalize((0.4913997, 0.4821584, 0.4465309),
                                 (0.2470322, 0.2434851, 0.2615878))
            ])
    }

    # config = {'train': True, 'extra': True, 'test': False}
    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.CIFAR10(root=dataset_paths[i], transform=transf[i],
                        train=config[i], download=True) for i in config.keys()}

    # print(datasets['train'][0])
    # exit()



    datasets['train'].targets = torch.LongTensor(datasets['train'].targets)
    datasets['test'].targets = torch.LongTensor(datasets['test'].targets)




    # split train into train and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].targets,
        n_classes=10,
        n_samples_per_class=np.repeat(500, 10)) # 500 per class


    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4913997, 0.4821584, 0.4465309),
                                     (0.2470322, 0.2434851, 0.2615878)),
                ])


    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4913997, 0.4821584, 0.4465309),
                                     (0.2470322, 0.2434851, 0.2615878)),
                ])

    # transf['train_'] = transf['train']
    # transf['valid'] = transf['train']



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


    dataloaders = {i: DataLoader(datasets[i], num_workers=args.num_workers, pin_memory=True,
        batch_size=args.batch_size, shuffle=config[i]) for i in config.keys()}


    return dataloaders
