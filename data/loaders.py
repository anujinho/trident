import os
import pickle

import learn2learn as l2l
import numpy as np
import torch
import torchvision
from learn2learn.data.utils import (download_file,
                                    download_file_from_google_drive)
from torch.utils.data import ConcatDataset, Dataset


class Omniglotmix(Dataset):
    def __init__(self, root, download=False, transform=None, target_transforms=None):
        """ Dataset class for the Omniglot dataset including the background and evaluation classes
        # Arguments: 
            root: root folder to fetch/download the datasets from/at
            transforms: transforms for the image before fetching
            target_transforms: transforms for the class numbers
        """
        self.root = root
        self.transforms = transform
        self.target_transforms = target_transforms
        bg = torchvision.datasets.omniglot.Omniglot(
            background=True, root=self.root, download=download)
        eval = torchvision.datasets.omniglot.Omniglot(
            background=False, root=self.root, download=download, target_transform=lambda x: x+964)
        # target_transform the labels of eval before concatting since they would overwrite the bg labels (bg has 964 classes)
        # add other unlabeled datasets here for unsupervised/semi-supervised few-shot
        self.dataset = ConcatDataset((bg, eval))
        self._bookkeeping_path = os.path.join(
            self.root, 'omniglot-bookkeeping.pkl')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, char_class = self.dataset[index]
        if self.transforms:
            image = self.transforms(image)

        if self.target_transforms:
            char_class = self.target_transforms(char_class)

        return image, char_class


def download_pkl(google_drive_id, data_root, mode):
    filename = 'mini-imagenet-cache-' + mode
    file_path = os.path.join(data_root, filename)

    if not os.path.exists(file_path + '.pkl'):
        print('Downloading:', file_path + '.pkl')
        download_file_from_google_drive(google_drive_id, file_path + '.pkl')
    else:
        print("Data was already downloaded")


def index_classes(items):
    idx = {}
    for i in items:
        if (i not in idx):
            idx[i] = len(idx)
    return idx


class MiniImageNet(Dataset):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/mini_imagenet.py)
    **Description**
    The *mini*-ImageNet dataset was originally introduced by Vinyals et al., 2016.
    It consists of 60'000 colour images of sizes 84x84 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
    The classes are sampled from the ImageNet dataset, and we use the splits from Ravi & Larochelle, 2017.
    **References**
    1. Vinyals et al. 2016. “Matching Networks for One Shot Learning.” NeurIPS.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.
    **Arguments**
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    """

    def __init__(self,
                 root,
                 mode,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(MiniImageNet, self).__init__()
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self._bookkeeping_path = os.path.join(
            self.root, 'mini-imagenet-bookkeeping-' + mode + '.pkl')
        if self.mode == 'test':
            google_drive_file_id = '1wpmY-hmiJUUlRBkO9ZDCXAcIpHEFdOhD'
            dropbox_file_link = 'https://www.dropbox.com/s/ye9jeb5tyz0x01b/mini-imagenet-cache-test.pkl?dl=1'
        elif self.mode == 'train':
            google_drive_file_id = '1I3itTXpXxGV68olxM5roceUMG8itH9Xj'
            dropbox_file_link = 'https://www.dropbox.com/s/9g8c6w345s2ek03/mini-imagenet-cache-train.pkl?dl=1'
        elif self.mode == 'validation':
            google_drive_file_id = '1KY5e491bkLFqJDp0-UWou3463Mo8AOco'
            dropbox_file_link = 'https://www.dropbox.com/s/ip1b7se3gij3r1b/mini-imagenet-cache-validation.pkl?dl=1'
        else:
            raise ('ValueError', 'Needs to be train, test or validation')

        pickle_file = os.path.join(
            self.root, 'mini-imagenet-cache-' + mode + '.pkl')
        try:
            if not self._check_exists() and download:
                print('Downloading mini-ImageNet --', mode)
                download_pkl(google_drive_file_id, self.root, mode)
            with open(pickle_file, 'rb') as f:
                self.data = pickle.load(f)
        except pickle.UnpicklingError:
            if not self._check_exists() and download:
                print('Download failed. Re-trying mini-ImageNet --', mode)
                download_file(dropbox_file_link, pickle_file)
            with open(pickle_file, 'rb') as f:
                self.data = pickle.load(f)

        self.x = torch.from_numpy(
            self.data["image_data"]).permute(0, 3, 1, 2).float()
        self.y = np.ones(len(self.x))

        self.class_idx = index_classes(self.data['class_dict'].keys())
        for class_name, idxs in self.data['class_dict'].items():
            for idx in idxs:
                self.y[idx] = self.class_idx[class_name]

    def __getitem__(self, idx):
        data = self.x[idx]
        if self.transform:
            data = self.transform(data)
        return data, self.y[idx]

    def __len__(self):
        return len(self.x)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'mini-imagenet-cache-' + self.mode + '.pkl'))
