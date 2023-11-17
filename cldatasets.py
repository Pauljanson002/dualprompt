# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

from itertools import chain
import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask,main_buffer = split_single_dataset(dataset_train, dataset_val, args)
    elif args.dataset == "cub":
        dataset_train, dataset_val = get_dataset('CUB200', transform_train, transform_val, args)
        args.nb_classes = len(dataset_val.classes)
        splited_datasets, class_mask,main_buffer = split_single_dataset(dataset_train, dataset_val, args)
    elif args.dataset == "aircraft":
        dataset_train, dataset_val = get_dataset('aircraft', transform_train, transform_val, args)
        args.nb_classes = 100
        args.num_tasks = 10        
        splited_datasets, class_mask,main_buffer = split_single_dataset(dataset_train, dataset_val, args)
    elif args.dataset == "cars":
        dataset_train, dataset_val = get_dataset('cars', transform_train, transform_val, args)
        args.nb_classes = 190
        args.num_tasks = 10
        splited_datasets, class_mask,main_buffer = split_single_dataset(dataset_train, dataset_val, args)
    elif args.dataset == "country":
        dataset_train , dataset_val = get_dataset('country', transform_train, transform_val, args)
        splited_datasets, class_mask,main_buffer = split_single_dataset(dataset_train, dataset_val, args)
    elif args.dataset == "gtsrb":
        dataset_train , dataset_val = get_dataset('gtsrb', transform_train, transform_val, args)
        args.nb_classes = 40
        splited_datasets, class_mask,main_buffer = split_single_dataset(dataset_train, dataset_val, args)
    elif args.dataset == "birdsnap":
        dataset_train, dataset_val = get_dataset('birdsnap', transform_train, transform_val, args)
        # args.nb_classes = len(dataset_val.classes)
        splited_datasets, class_mask,main_buffer = split_single_dataset(dataset_train, dataset_val, args)
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle:
            random.shuffle(dataset_list)
        print(dataset_list)
    
        args.nb_classes = 0
    
    full_dataset_train = dataset_train

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]
        elif args.dataset == "cub" or args.dataset == "cars" or args.dataset == "aircraft" or args.dataset == "country" or args.dataset == "gtsrb" or args.dataset == "birdsnap":
            dataset_train, dataset_val = splited_datasets[i]
        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
        
        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        def collate_fn(examples):
            images = []
            labels = []
            for example in examples:
                if example[0].shape[0] == 3:
                    images.append(example[0])
                    labels.append(example[1])
            pixel_values = torch.stack(images)
            labels = torch.tensor(labels)
            return pixel_values, labels
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=collate_fn,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=collate_fn,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask,main_buffer, full_dataset_train



def get_classnames(args):
    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)
    if args.dataset.startswith('Split-'):
        ds,_ = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)
        return ds.classes
    elif args.dataset == "cub":
        ds,_ = get_dataset('CUB200', transform_train, transform_val, args)
        return ds.classes
    elif args.dataset == "cars":
        ds,_ = get_dataset('cars', transform_train, transform_val, args)
        return ds.classes
    elif args.dataset == "aircraft":
        ds,_ = get_dataset('aircraft', transform_train, transform_val, args)
        return ds.classes
    elif args.dataset == "country":
        ds,_ = get_dataset('country', transform_train, transform_val, args)
        return ds.classes
    elif args.dataset == "gtsrb":
        ds,_ = get_dataset('gtsrb', transform_train, transform_val, args)
        return ds.classes
    elif args.dataset == "birdsnap":
        ds,_ = get_dataset('birdsnap', transform_train, transform_val, args)
        return ds.classes
    else:
        NotImplementedError()

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    elif dataset == 'cars':
        dataset_train = Cars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Cars(args.data_path, split='test', download=True, transform=transform_val)
    elif dataset == 'aircraft':
        dataset_train = Aircraft(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Aircraft(args.data_path, split='test', download=True, transform=transform_val)
    elif dataset == "country":
        dataset_train = Country(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Country(args.data_path, split='test', download=True, transform=transform_val)
    elif dataset == "gtsrb":
        dataset_train = GTSRB(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = GTSRB(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = CUB(args.data_path, train=False, download=True, transform=transform_val)
    elif dataset == "birdsnap":
        dataset_train = Birdsnap(args.data_path, split="train", download=True, transform=transform_train)
        dataset_val = Birdsnap(args.data_path, split="test", download=True, transform=transform_val)
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data
    
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    if args.dataset == "cars":
        nb_classes = len(dataset_val.classes) - 6
    elif args.dataset == "gtsrb":
        nb_classes = 40
    else:
        nb_classes = args.nb_classes
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()
    
    if args.shuffle:
        random.shuffle(labels)
    buffer = [[] for _ in range(args.nb_classes)]
    main_buffer = []
    for t in range(args.num_tasks):
        
         
        train_split_indices = []
        test_split_indices = []
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)

                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        buffer_per_class = args.memory // (classes_per_task* (t+1))

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                buffer[int(dataset_train.targets[k])].append(k)

        for i in range(len(buffer)):
            random.shuffle(buffer[i])
            buffer[i] = buffer[i][:buffer_per_class]
            print(f"buffer[{i}] : {len(buffer[i])}")
        buffer_flat = list(chain.from_iterable(buffer))
        main_buffer.append(buffer_flat)
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)
        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask,main_buffer

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)