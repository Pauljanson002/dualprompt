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
        # ds,_ = get_dataset('birdsnap', transform_train, transform_val, args)
        classes = [
    'Acadian Flycatcher',
    'Acorn Woodpecker',
    'Alder Flycatcher',
    'Allens Hummingbird',
    'Altamira Oriole',
    'American Avocet',
    'American Bittern',
    'American Black Duck',
    'American Coot',
    'American Crow',
    'American Dipper',
    'American Golden Plover',
    'American Goldfinch',
    'American Kestrel',
    'American Oystercatcher',
    'American Pipit',
    'American Redstart',
    'American Robin',
    'American Three toed Woodpecker',
    'American Tree Sparrow',
    'American White Pelican',
    'American Wigeon',
    'American Woodcock',
    'Anhinga',
    'Annas Hummingbird',
    'Arctic Tern',
    'Ash throated Flycatcher',
    'Audubons Oriole',
    'Bairds Sandpiper',
    'Bald Eagle',
    'Baltimore Oriole',
    'Band tailed Pigeon',
    'Barn Swallow',
    'Barred Owl',
    'Barrows Goldeneye',
    'Bay breasted Warbler',
    'Bells Vireo',
    'Belted Kingfisher',
    'Bewicks Wren',
    'Black Guillemot',
    'Black Oystercatcher',
    'Black Phoebe',
    'Black Rosy Finch',
    'Black Scoter',
    'Black Skimmer',
    'Black Tern',
    'Black Turnstone',
    'Black Vulture',
    'Black and white Warbler',
    'Black backed Woodpecker',
    'Black bellied Plover',
    'Black billed Cuckoo',
    'Black billed Magpie',
    'Black capped Chickadee',
    'Black chinned Hummingbird',
    'Black chinned Sparrow',
    'Black crested Titmouse',
    'Black crowned Night Heron',
    'Black headed Grosbeak',
    'Black legged Kittiwake',
    'Black necked Stilt',
    'Black throated Blue Warbler',
    'Black throated Gray Warbler',
    'Black throated Green Warbler',
    'Black throated Sparrow',
    'Blackburnian Warbler',
    'Blackpoll Warbler',
    'Blue Grosbeak',
    'Blue Jay',
    'Blue gray Gnatcatcher',
    'Blue headed Vireo',
    'Blue winged Teal',
    'Blue winged Warbler',
    'Boat tailed Grackle',
    'Bobolink',
    'Bohemian Waxwing',
    'Bonapartes Gull',
    'Boreal Chickadee',
    'Brandts Cormorant',
    'Brant',
    'Brewers Blackbird',
    'Brewers Sparrow',
    'Bridled Titmouse',
    'Broad billed Hummingbird',
    'Broad tailed Hummingbird',
    'Broad winged Hawk',
    'Bronzed Cowbird',
    'Brown Creeper',
    'Brown Pelican',
    'Brown Thrasher',
    'Brown capped Rosy Finch',
    'Brown crested Flycatcher',
    'Brown headed Cowbird',
    'Brown headed Nuthatch',
    'Bufflehead',
    'Bullocks Oriole',
    'Burrowing Owl',
    'Bushtit',
    'Cackling Goose',
    'Cactus Wren',
    'California Gull',
    'California Quail',
    'California Thrasher',
    'California Towhee',
    'Calliope Hummingbird',
    'Canada Goose',
    'Canada Warbler',
    'Canvasback',
    'Canyon Towhee',
    'Canyon Wren',
    'Cape May Warbler',
    'Carolina Chickadee',
    'Carolina Wren',
    'Caspian Tern',
    'Cassins Finch',
    'Cassins Kingbird',
    'Cassins Sparrow',
    'Cassins Vireo',
    'Cattle Egret',
    'Cave Swallow',
    'Cedar Waxwing',
    'Cerulean Warbler',
    'Chestnut backed Chickadee',
    'Chestnut collared Longspur',
    'Chestnut sided Warbler',
    'Chihuahuan Raven',
    'Chimney Swift',
    'Chipping Sparrow',
    'Cinnamon Teal',
    'Clapper Rail',
    'Clarks Grebe',
    'Clarks Nutcracker',
    'Clay colored Sparrow',
    'Cliff Swallow',
    'Common Black Hawk',
    'Common Eider',
    'Common Gallinule',
    'Common Goldeneye',
    'Common Grackle',
    'Common Ground Dove',
    'Common Loon',
    'Common Merganser',
    'Common Murre',
    'Common Nighthawk',
    'Common Raven',
    'Common Redpoll',
    'Common Tern',
    'Common Yellowthroat',
    'Connecticut Warbler',
    'Coopers Hawk',
    'Cordilleran Flycatcher',
    'Costas Hummingbird',
    'Couchs Kingbird',
    'Crested Caracara',
    'Curve billed Thrasher',
    'Dark eyed Junco',
    'Dickcissel',
    'Double crested Cormorant',
    'Downy Woodpecker',
    'Dunlin',
    'Dusky Flycatcher',
    'Dusky Grouse',
    'Eared Grebe',
    'Eastern Bluebird',
    'Eastern Kingbird',
    'Eastern Meadowlark',
    'Eastern Phoebe',
    'Eastern Screech Owl',
    'Eastern Towhee',
    'Eastern Wood Pewee',
    'Elegant Trogon',
    'Elf Owl',
    'Eurasian Collared Dove',
    'Eurasian Wigeon',
    'European Starling',
    'Evening Grosbeak',
    'Ferruginous Hawk',
    'Ferruginous Pygmy Owl',
    'Field Sparrow',
    'Fish Crow',
    'Florida Scrub Jay',
    'Forsters Tern',
    'Fox Sparrow',
    'Franklins Gull',
    'Fulvous Whistling Duck',
    'Gadwall',
    'Gambels Quail',
    'Gila Woodpecker',
    'Glaucous Gull',
    'Glaucous winged Gull',
    'Glossy Ibis',
    'Golden Eagle',
    'Golden crowned Kinglet',
    'Golden crowned Sparrow',
    'Golden fronted Woodpecker',
    'Golden winged Warbler',
    'Grasshopper Sparrow',
    'Gray Catbird',
    'Gray Flycatcher',
    'Gray Jay',
    'Gray Kingbird',
    'Gray cheeked Thrush',
    'Gray crowned Rosy Finch',
    'Great Black backed Gull',
    'Great Blue Heron',
    'Great Cormorant',
    'Great Crested Flycatcher',
    'Great Egret',
    'Great Gray Owl',
    'Great Horned Owl',
    'Great Kiskadee',
    'Great tailed Grackle',
    'Greater Prairie Chicken',
    'Greater Roadrunner',
    'Greater Sage Grouse',
    'Greater Scaup',
    'Greater White fronted Goose',
    'Greater Yellowlegs',
    'Green Jay',
    'Green tailed Towhee',
    'Green winged Teal',
    'Groove billed Ani',
    'Gull billed Tern',
    'Hairy Woodpecker',
    'Hammonds Flycatcher',
    'Harlequin Duck',
    'Harriss Hawk',
    'Harriss Sparrow',
    'Heermanns Gull',
    'Henslows Sparrow',
    'Hepatic Tanager',
    'Hermit Thrush',
    'Herring Gull',
    'Hoary Redpoll',
    'Hooded Merganser',
    'Hooded Oriole',
    'Hooded Warbler',
    'Horned Grebe',
    'Horned Lark',
    'House Finch',
    'House Sparrow',
    'House Wren',
    'Huttons Vireo',
    'Iceland Gull',
    'Inca Dove',
    'Indigo Bunting',
    'Killdeer',
    'King Rail',
    'Ladder backed Woodpecker',
    'Lapland Longspur',
    'Lark Bunting',
    'Lark Sparrow',
    'Laughing Gull',
    'Lazuli Bunting',
    'Le Contes Sparrow',
    'Least Bittern',
    'Least Flycatcher',
    'Least Grebe',
    'Least Sandpiper',
    'Least Tern',
    'Lesser Goldfinch',
    'Lesser Nighthawk',
    'Lesser Scaup',
    'Lesser Yellowlegs',
    'Lewiss Woodpecker',
    'Limpkin',
    'Lincolns Sparrow',
    'Little Blue Heron',
    'Loggerhead Shrike',
    'Long billed Curlew',
    'Long billed Dowitcher',
    'Long billed Thrasher',
    'Long eared Owl',
    'Long tailed Duck',
    'Louisiana Waterthrush',
    'Magnificent Frigatebird',
    'Magnolia Warbler',
    'Mallard',
    'Marbled Godwit',
    'Marsh Wren',
    'Merlin',
    'Mew Gull',
    'Mexican Jay',
    'Mississippi Kite',
    'Monk Parakeet',
    'Mottled Duck',
    'Mountain Bluebird',
    'Mountain Chickadee',
    'Mountain Plover',
    'Mourning Dove',
    'Mourning Warbler',
    'Muscovy Duck',
    'Mute Swan',
    'Nashville Warbler',
    'Nelsons Sparrow',
    'Neotropic Cormorant',
    'Northern Bobwhite',
    'Northern Cardinal',
    'Northern Flicker',
    'Northern Gannet',
    'Northern Goshawk',
    'Northern Harrier',
    'Northern Hawk Owl',
    'Northern Mockingbird',
    'Northern Parula',
    'Northern Pintail',
    'Northern Rough winged Swallow',
    'Northern Saw whet Owl',
    'Northern Shrike',
    'Northern Waterthrush',
    'Nuttalls Woodpecker',
    'Oak Titmouse',
    'Olive Sparrow',
    'Olive sided Flycatcher',
    'Orange crowned Warbler',
    'Orchard Oriole',
    'Osprey',
    'Ovenbird',
    'Pacific Golden Plover',
    'Pacific Loon',
    'Pacific Wren',
    'Pacific slope Flycatcher',
    'Painted Bunting',
    'Painted Redstart',
    'Palm Warbler',
    'Pectoral Sandpiper',
    'Peregrine Falcon',
    'Phainopepla',
    'Philadelphia Vireo',
    'Pied billed Grebe',
    'Pigeon Guillemot',
    'Pileated Woodpecker',
    'Pine Grosbeak',
    'Pine Siskin',
    'Pine Warbler',
    'Piping Plover',
    'Plumbeous Vireo',
    'Prairie Falcon',
    'Prairie Warbler',
    'Prothonotary Warbler',
    'Purple Finch',
    'Purple Gallinule',
    'Purple Martin',
    'Purple Sandpiper',
    'Pygmy Nuthatch',
    'Pyrrhuloxia',
    'Red Crossbill',
    'Red Knot',
    'Red Phalarope',
    'Red bellied Woodpecker',
    'Red breasted Merganser',
    'Red breasted Nuthatch',
    'Red breasted Sapsucker',
    'Red cockaded Woodpecker',
    'Red eyed Vireo',
    'Red headed Woodpecker',
    'Red naped Sapsucker',
    'Red necked Grebe',
    'Red necked Phalarope',
    'Red shouldered Hawk',
    'Red tailed Hawk',
    'Red throated Loon',
    'Red winged Blackbird',
    'Reddish Egret',
    'Redhead',
    'Ring billed Gull',
    'Ring necked Duck',
    'Ring necked Pheasant',
    'Rock Pigeon',
    'Rock Ptarmigan',
    'Rock Sandpiper',
    'Rock Wren',
    'Rose breasted Grosbeak',
    'Roseate Tern',
    'Rosss Goose',
    'Rough legged Hawk',
    'Royal Tern',
    'Ruby crowned Kinglet',
    'Ruby throated Hummingbird',
    'Ruddy Duck',
    'Ruddy Turnstone',
    'Ruffed Grouse',
    'Rufous Hummingbird',
    'Rufous crowned Sparrow',
    'Rusty Blackbird',
    'Sage Thrasher',
    'Saltmarsh Sparrow',
    'Sanderling',
    'Sandhill Crane',
    'Sandwich Tern',
    'Says Phoebe',
    'Scaled Quail',
    'Scarlet Tanager',
    'Scissor tailed Flycatcher',
    'Scotts Oriole',
    'Seaside Sparrow',
    'Sedge Wren',
    'Semipalmated Plover',
    'Semipalmated Sandpiper',
    'Sharp shinned Hawk',
    'Sharp tailed Grouse',
    'Short billed Dowitcher',
    'Short eared Owl',
    'Snail Kite',
    'Snow Bunting',
    'Snow Goose',
    'Snowy Egret',
    'Snowy Owl',
    'Snowy Plover',
    'Solitary Sandpiper',
    'Song Sparrow',
    'Sooty Grouse',
    'Sora',
    'Spotted Owl',
    'Spotted Sandpiper',
    'Spotted Towhee',
    'Spruce Grouse',
    'Stellers Jay',
    'Stilt Sandpiper',
    'Summer Tanager',
    'Surf Scoter',
    'Surfbird',
    'Swainsons Hawk',
    'Swainsons Thrush',
    'Swallow tailed Kite',
    'Swamp Sparrow',
    'Tennessee Warbler',
    'Thayers Gull',
    'Townsends Solitaire',
    'Townsends Warbler',
    'Tree Swallow',
    'Tricolored Heron',
    'Tropical Kingbird',
    'Trumpeter Swan',
    'Tufted Titmouse',
    'Tundra Swan',
    'Turkey Vulture',
    'Upland Sandpiper',
    'Varied Thrush',
    'Veery',
    'Verdin',
    'Vermilion Flycatcher',
    'Vesper Sparrow',
    'Violet green Swallow',
    'Virginia Rail',
    'Wandering Tattler',
    'Warbling Vireo',
    'Western Bluebird',
    'Western Grebe',
    'Western Gull',
    'Western Kingbird',
    'Western Meadowlark',
    'Western Sandpiper',
    'Western Screech Owl',
    'Western Scrub Jay',
    'Western Tanager',
    'Western Wood Pewee',
    'Whimbrel',
    'White Ibis',
    'White breasted Nuthatch',
    'White crowned Sparrow',
    'White eyed Vireo',
    'White faced Ibis',
    'White headed Woodpecker',
    'White rumped Sandpiper',
    'White tailed Hawk',
    'White tailed Kite',
    'White tailed Ptarmigan',
    'White throated Sparrow',
    'White throated Swift',
    'White winged Crossbill',
    'White winged Dove',
    'White winged Scoter',
    'Wild Turkey',
    'Willet',
    'Williamsons Sapsucker',
    'Willow Flycatcher',
    'Willow Ptarmigan',
    'Wilsons Phalarope',
    'Wilsons Plover',
    'Wilsons Snipe',
    'Wilsons Warbler',
    'Winter Wren',
    'Wood Stork',
    'Wood Thrush',
    'Worm eating Warbler',
    'Wrentit',
    'Yellow Warbler',
    'Yellow bellied Flycatcher',
    'Yellow bellied Sapsucker',
    'Yellow billed Cuckoo',
    'Yellow billed Magpie',
    'Yellow breasted Chat',
    'Yellow crowned Night Heron',
    'Yellow eyed Junco',
    'Yellow headed Blackbird',
    'Yellow rumped Warbler',
    'Yellow throated Vireo',
    'Yellow throated Warbler',
    'Zone tailed Hawk',
]
        return classes
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