# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import os

import os.path
import pathlib
from pathlib import Path

from typing import Any, Tuple

import glob
from shutil import move, rmtree

import numpy as np
import datasets as hfdatasets
import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive

import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .dataset_utils import read_image_file, read_label_file

class MNIST_RGB(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_RGB, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNIST(MNIST_RGB):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class NotMNIST(MNIST_RGB):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
        self.filename = 'notMNIST.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')


        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class Flowers102(datasets.Flowers102):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(Flowers102, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self.targets = []
        self._image_files = []
        for image_id in image_ids:
            self.targets.append(image_id_to_label[image_id] - 1) # -1 for 0-based indexing
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")
        self.classes = list(set(self.targets))
    
    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self.targets[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

class StanfordCars(datasets.StanfordCars):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super(StanfordCars, self).__init__(root, transform=transform, target_transform=target_transform, download=download)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()
    


class Cars(torch.utils.data.Dataset):
    def __init__(self,root, split="train", transform=None, target_transform=None, download=False):
        if split == "train":
            self.hfdata = hfdatasets.load_dataset("Multimodal-Fatima/StanfordCars_train",cache_dir=root)['train']
        else:
            self.hfdata = hfdatasets.load_dataset("Multimodal-Fatima/StanfordCars_test",cache_dir=root)['test']
        self.transform = transform
        self.classes = self.hfdata.features['label'].names
        self.targets = self.hfdata['label']
        
    def __len__(self):
        return len(self.hfdata)
    def __getitem__(self, idx):
        img = self.hfdata[idx]['image']
        target = self.hfdata[idx]['label']
        if self.transform is not None:
            img = self.transform(img)
        return img,target

from torchvision.datasets import FGVCAircraft
class Aircraft(torch.utils.data.Dataset):
    def __init__(self,root, split="train", transform=None, target_transform=None, download=True):
        if split == "train":
            self.dataset = FGVCAircraft(root, split='trainval',transform=transform, download=True)
        else:
            self.dataset = FGVCAircraft(root, split='test',transform=transform, download=True)
        self.transform = transform
        self.targets = [self.dataset[i][1] for i in range(len(self.dataset))]

        self.classes = [
    '707-320',
    '727-200',
    '737-200',
    '737-300',
    '737-400',
    '737-500',
    '737-600',
    '737-700',
    '737-800',
    '737-900',
    '747-100',
    '747-200',
    '747-300',
    '747-400',
    '757-200',
    '757-300',
    '767-200',
    '767-300',
    '767-400',
    '777-200',
    '777-300',
    'A300B4',
    'A310',
    'A318',
    'A319',
    'A320',
    'A321',
    'A330-200',
    'A330-300',
    'A340-200',
    'A340-300',
    'A340-500',
    'A340-600',
    'A380',
    'ATR-42',
    'ATR-72',
    'An-12',
    'BAE 146-200',
    'BAE 146-300',
    'BAE-125',
    'Beechcraft 1900',
    'Boeing 717',
    'C-130',
    'C-47',
    'CRJ-200',
    'CRJ-700',
    'CRJ-900',
    'Cessna 172',
    'Cessna 208',
    'Cessna 525',
    'Cessna 560',
    'Challenger 600',
    'DC-10',
    'DC-3',
    'DC-6',
    'DC-8',
    'DC-9-30',
    'DH-82',
    'DHC-1',
    'DHC-6',
    'DHC-8-100',
    'DHC-8-300',
    'DR-400',
    'Dornier 328',
    'E-170',
    'E-190',
    'E-195',
    'EMB-120',
    'ERJ 135',
    'ERJ 145',
    'Embraer Legacy 600',
    'Eurofighter Typhoon',
    'F-16A/B',
    'F/A-18',
    'Falcon 2000',
    'Falcon 900',
    'Fokker 100',
    'Fokker 50',
    'Fokker 70',
    'Global Express',
    'Gulfstream IV',
    'Gulfstream V',
    'Hawk T1',
    'Il-76',
    'L-1011',
    'MD-11',
    'MD-80',
    'MD-87',
    'MD-90',
    'Metroliner',
    'Model B200',
    'PA-28',
    'SR-20',
    'Saab 2000',
    'Saab 340',
    'Spitfire',
    'Tornado',
    'Tu-134',
    'Tu-154',
    'Yak-42',
]


        

        
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        target = self.dataset[idx][1]
        
        return img,target


from torchvision.datasets import Country211 as Country


class Country211(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None, download=True):
        if split == "train":
            self.dataset = Country(root, split='train',transform=transform, download=download)
        else:
            self.dataset = Country(root, split='test',transform=transform, download=download)
        self.classes = [
    'Andorra',
    'United Arab Emirates',
    'Afghanistan',
    'Antigua and Barbuda',
    'Anguilla',
    'Albania',
    'Armenia',
    'Angola',
    'Antarctica',
    'Argentina',
    'Austria',
    'Australia',
    'Aruba',
    'Aland Islands',
    'Azerbaijan',
    'Bosnia and Herzegovina',
    'Barbados',
    'Bangladesh',
    'Belgium',
    'Burkina Faso',
    'Bulgaria',
    'Bahrain',
    'Benin',
    'Bermuda',
    'Brunei Darussalam',
    'Bolivia',
    'Bonaire, Saint Eustatius and Saba',
    'Brazil',
    'Bahamas',
    'Bhutan',
    'Botswana',
    'Belarus',
    'Belize',
    'Canada',
    'DR Congo',
    'Central African Republic',
    'Switzerland',
    "Cote d'Ivoire",
    'Cook Islands',
    'Chile',
    'Cameroon',
    'China',
    'Colombia',
    'Costa Rica',
    'Cuba',
    'Cabo Verde',
    'Curacao',
    'Cyprus',
    'Czech Republic',
    'Germany',
    'Denmark',
    'Dominica',
    'Dominican Republic',
    'Algeria',
    'Ecuador',
    'Estonia',
    'Egypt',
    'Spain',
    'Ethiopia',
    'Finland',
    'Fiji',
    'Falkland Islands',
    'Faeroe Islands',
    'France',
    'Gabon',
    'United Kingdom',
    'Grenada',
    'Georgia',
    'French Guiana',
    'Guernsey',
    'Ghana',
    'Gibraltar',
    'Greenland',
    'Gambia',
    'Guadeloupe',
    'Greece',
    'South Georgia and South Sandwich Is.',
    'Guatemala',
    'Guam',
    'Guyana',
    'Hong Kong',
    'Honduras',
    'Croatia',
    'Haiti',
    'Hungary',
    'Indonesia',
    'Ireland',
    'Israel',
    'Isle of Man',
    'India',
    'Iraq',
    'Iran',
    'Iceland',
    'Italy',
    'Jersey',
    'Jamaica',
    'Jordan',
    'Japan',
    'Kenya',
    'Kyrgyz Republic',
    'Cambodia',
    'St. Kitts and Nevis',
    'North Korea',
    'South Korea',
    'Kuwait',
    'Cayman Islands',
    'Kazakhstan',
    'Laos',
    'Lebanon',
    'St. Lucia',
    'Liechtenstein',
    'Sri Lanka',
    'Liberia',
    'Lithuania',
    'Luxembourg',
    'Latvia',
    'Libya',
    'Morocco',
    'Monaco',
    'Moldova',
    'Montenegro',
    'Saint-Martin',
    'Madagascar',
    'Macedonia',
    'Mali',
    'Myanmar',
    'Mongolia',
    'Macau',
    'Martinique',
    'Mauritania',
    'Malta',
    'Mauritius',
    'Maldives',
    'Malawi',
    'Mexico',
    'Malaysia',
    'Mozambique',
    'Namibia',
    'New Caledonia',
    'Nigeria',
    'Nicaragua',
    'Netherlands',
    'Norway',
    'Nepal',
    'New Zealand',
    'Oman',
    'Panama',
    'Peru',
    'French Polynesia',
    'Papua New Guinea',
    'Philippines',
    'Pakistan',
    'Poland',
    'Puerto Rico',
    'Palestine',
    'Portugal',
    'Palau',
    'Paraguay',
    'Qatar',
    'Reunion',
    'Romania',
    'Serbia',
    'Russia',
    'Rwanda',
    'Saudi Arabia',
    'Solomon Islands',
    'Seychelles',
    'Sudan',
    'Sweden',
    'Singapore',
    'St. Helena',
    'Slovenia',
    'Svalbard and Jan Mayen Islands',
    'Slovakia',
    'Sierra Leone',
    'San Marino',
    'Senegal',
    'Somalia',
    'South Sudan',
    'El Salvador',
    'Sint Maarten',
    'Syria',
    'Eswatini',
    'Togo',
    'Thailand',
    'Tajikistan',
    'Timor-Leste',
    'Turkmenistan',
    'Tunisia',
    'Tonga',
    'Turkey',
    'Trinidad and Tobago',
    'Taiwan',
    'Tanzania',
    'Ukraine',
    'Uganda',
    'United States',
    'Uruguay',
    'Uzbekistan',
    'Vatican',
    'Venezuela',
    'British Virgin Islands',
    'United States Virgin Islands',
    'Vietnam',
    'Vanuatu',
    'Samoa',
    'Kosovo',
    'Yemen',
    'South Africa',
    'Zambia',
    'Zimbabwe',
]
    



    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        target = self.dataset[idx][1]
        
        return img,target

from torchvision.datasets import GTSRB as GTS
class GTSRB(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None, download=True):
        if split == "train":
            self.dataset = GTS(root, split='train',transform=transform, download=download)
        else:
            self.dataset = GTS(root, split='test',transform=transform, download=download)
        self.targets = [self.dataset._samples[i][1] for i in range(len(self.dataset))]
        self.classes = [
    'red and white circle 20 kph speed limit',
    'red and white circle 30 kph speed limit',
    'red and white circle 50 kph speed limit',
    'red and white circle 60 kph speed limit',
    'red and white circle 70 kph speed limit',
    'red and white circle 80 kph speed limit',
    'end / de-restriction of 80 kph speed limit',
    'red and white circle 100 kph speed limit',
    'red and white circle 120 kph speed limit',
    'red and white circle red car and black car no passing',
    'red and white circle red truck and black car no passing',
    'red and white triangle road intersection warning',
    'white and yellow diamond priority road',
    'red and white upside down triangle yield right-of-way',
    'stop',
    'empty red and white circle',
    'red and white circle no truck entry',
    'red circle with white horizonal stripe no entry',
    'red and white triangle with exclamation mark warning',
    'red and white triangle with black left curve approaching warning',
    'red and white triangle with black right curve approaching warning',
    'red and white triangle with black double curve approaching warning',
    'red and white triangle rough / bumpy road warning',
    'red and white triangle car skidding / slipping warning',
    'red and white triangle with merging / narrow lanes warning',
    'red and white triangle with person digging / construction / road work warning',
    'red and white triangle with traffic light approaching warning',
    'red and white triangle with person walking warning',
    'red and white triangle with child and person walking warning',
    'red and white triangle with bicyle warning',
    'red and white triangle with snowflake / ice warning',
    'red and white triangle with deer warning',
    'white circle with gray strike bar no speed limit',
    'blue circle with white right turn arrow mandatory',
    'blue circle with white left turn arrow mandatory',
    'blue circle with white forward arrow mandatory',
    'blue circle with white forward or right turn arrow mandatory',
    'blue circle with white forward or left turn arrow mandatory',
    'blue circle with white keep right arrow mandatory',
    'blue circle with white keep left arrow mandatory',
    'blue circle with white arrows indicating a traffic circle',
    'white circle with gray strike bar indicating no passing for cars has ended',
    'white circle with gray strike bar indicating no passing for trucks has ended',
]
    



    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        target = self.dataset[idx][1]
        
        return img,target

from torchvision.datasets import ImageFolder
class Birdsnap(torch.utils.data.Dataset):
    def __init__(self,root="/home/paulj/data/birdsnap/download/images",split="train",transform=None,target_transform=None,download=False):
        self.root = "/home/paulj/data/birdsnap/download/images"
        self.set = ImageFolder("/home/paulj/data/birdsnap/download/images")
        self.idx_dir = "/home/paulj/data/birdsnap/"
        with open(f'{self.idx_dir}/test_images.txt','r') as f:
            lines = f.readlines()
            test_paths = [os.path.join(self.root,line.strip()) for line in lines if 'jpg' in line]
        test_indices = [idx for idx, ( path,_) in enumerate(self.set.samples) if path in test_paths]
        self.targets = np.array(self.set.targets)
        train_indices = [idx for idx in range(len(self.set)) if idx not in test_indices]
        if split == "train":
            self.set = torch.utils.data.Subset(self.set,train_indices)
        else:
            self.set = torch.utils.data.Subset(self.set,test_indices)
        self.targets = self.targets[train_indices] if split == "train" else self.targets[test_indices]
        self.transform = transform
        self.classes = [
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
    

    def __len__(self):
        return len(self.set)
    def __getitem__(self, idx):
        img = self.set[idx][0]
        target = self.set[idx][1]
        if self.transform is not None:
            img = self.transform(img)
        return img,target


class CUB200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://data.deepai.org/CUB200(2011).zip'
        self.filename = 'CUB200(2011).zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'CUB_200_2011')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(root)
            zip_ref.close()

            import tarfile
            tar_ref = tarfile.open(os.path.join(root, 'CUB_200_2011.tgz'), 'r')
            tar_ref.extractall(root)
            tar_ref.close()

            self.split()
        
        if self.train:
            fpath = os.path.join(root, 'CUB_200_2011', 'train')

        else:
            fpath = os.path.join(root, 'CUB_200_2011', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.root + 'CUB_200_2011/train'
        test_folder = self.root + 'CUB_200_2011/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        images = self.root + 'CUB_200_2011/images.txt'
        train_test_split = self.root + 'CUB_200_2011/train_test_split.txt'

        with open(images, 'r') as image:
            with open(train_test_split, 'r') as f:
                for line in f:
                    image_path = image.readline().split(' ')[-1]
                    image_path = image_path.replace('\n', '')
                    class_name = image_path.split('/')[0].split(' ')[-1]
                    src = self.root + 'CUB_200_2011/images/' + image_path

                    if line.split(' ')[-1].replace('\n', '') == '1':
                        if not os.path.exists(train_folder + '/' + class_name):
                            os.mkdir(train_folder + '/' + class_name)
                        dst = train_folder + '/' + image_path
                    else:
                        if not os.path.exists(test_folder + '/' + class_name):
                            os.mkdir(test_folder + '/' + class_name)
                        dst = test_folder + '/' + image_path
                    
                    move(src, dst)

class CUB(torch.utils.data.Dataset):
    def __init__(self,root, train=True, transform=None, target_transform=None, download=False):
        self.data = hfdatasets.load_dataset("alkzar90/CC6204-Hackaton-Cub-Dataset",cache_dir=root,split='train')
        self.transform = transform
        self.classes = self.data.features['label'].names
        self.targets = self.data['label']
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]['image']
        if self.transform is not None:
            img = self.transform(img)
        return img, self.data[idx]['label']


class TinyImagenet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.filename = 'tiny-imagenet-200.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)
        
        if not os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(os.path.join(root))
            zip_ref.close()

            self.split()

        if self.train:
            fpath = root + 'tiny-imagenet-200/train'

        else:
            fpath = root + 'tiny-imagenet-200/test'
        
        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        test_folder = self.root + 'tiny-imagenet-200/test'

        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(test_folder)

        val_dict = {}
        with open(self.root + 'tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]
                
        paths = glob.glob(self.root + 'tiny-imagenet-200/val/images/*')
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(test_folder + '/' + folder):
                os.mkdir(test_folder + '/' + folder)
                os.mkdir(test_folder + '/' + folder + '/images')
            
            
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            src = path
            dst = test_folder + '/' + folder + '/images/' + file
            move(src, dst)
        
        rmtree(self.root + 'tiny-imagenet-200/val')

class Scene67(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        image_url = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
        train_annos_url = 'http://web.mit.edu/torralba/www/TrainImages.txt'
        test_annos_url = 'http://web.mit.edu/torralba/www/TestImages.txt'
        urls = [image_url, train_annos_url, test_annos_url]
        image_fname = 'indoorCVPR_09.tar'
        self.train_annos_fname = 'TrainImage.txt'
        self.test_annos_fname = 'TestImage.txt'
        fnames = [image_fname, self.train_annos_fname, self.test_annos_fname]

        for url, fname in zip(urls, fnames):
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from ' + url)
                    download_url(url, root, filename=fname)
        if not os.path.exists(os.path.join(root, 'Scene67')):
            import tarfile
            with tarfile.open(os.path.join(root, image_fname)) as tar:
                tar.extractall(os.path.join(root, 'Scene67'))

            self.split()

        if self.train:
            fpath = os.path.join(root, 'Scene67', 'train')

        else:
            fpath = os.path.join(root, 'Scene67', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'train')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'train'))
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'test')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'test'))
        
        train_annos_file = os.path.join(self.root, self.train_annos_fname)
        test_annos_file = os.path.join(self.root, self.test_annos_fname)

        with open(train_annos_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'train/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0]))
                move(src, dst)
        
        with open(test_annos_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'test/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0]))
                move(src, dst)

class Imagenet_R(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
        self.filename = 'imagenet-r.tar'

        self.fpath = os.path.join(root, 'imagenet-r')
        if not os.path.isfile(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'imagenet-r')):
            import tarfile
            tar_ref = tarfile.open(os.path.join(root, self.filename), 'r')
            tar_ref.extractall(root)
            tar_ref.close()
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.dataset = datasets.ImageFolder(self.fpath, transform=transform)
            
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices
    
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]

            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'

        else:
            fpath = self.fpath + '/test'

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)