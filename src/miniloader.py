import os
import io
import json
import pickle
import tarfile
import shutil
import zipfile
import requests
from PIL import Image
from collections import defaultdict

import numpy as np
import scipy.io
import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from learn2learn.data.utils import (download_file,
                                    download_file_from_google_drive)
from torch.utils.data import ConcatDataset, Dataset


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
    Consists of 60'000 colour images of sizes 84x84 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
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

if __name__ == '__main__':
    import torchvision as tv
    from torchvision import transforms
    import learn2learn as l2l

    # train_dataset = CUBirds200(root='../dataset', mode='test')
    # train_dataset = l2l.data.MetaDataset(train_dataset)
    # tasks = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)
    #image_trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = MiniImageNet(root='../dataset/mini_imagenet', mode='test')
    data = l2l.data.MetaDataset(train_dataset)
    trans = [
                l2l.data.transforms.NWays(data, n=5),
                l2l.data.transforms.KShots(data, k=3),
                l2l.data.transforms.LoadData(data),
                #l2l.data.transforms.RemapLabels(data),
                l2l.data.transforms.ConsecutiveLabels(data)
            ]
    tasks = l2l.data.TaskDataset(dataset=data, task_transforms=trans, num_tasks=600)
    a = tasks.sample()
    print(a[1])
    for img in a[0]:
        plt.imshow(img.permute(1, 2, 0)/255.0)
        plt.show()

    