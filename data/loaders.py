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
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
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

class TieredImagenet(Dataset):
    
    """
    Like *mini*-ImageNet, *tiered*-ImageNet builds on top of ILSVRC-12, but consists of 608 classes (779,165 images) instead of 100.
    The train-validation-test split is made such that classes from similar categories are in the same splits.
    There are 34 categories each containing between 10 and 30 classes.
    Of these categories, 20 (351 classes; 448,695 images) are used for training,
    6 (97 classes; 124,261 images) for validation, and 8 (160 class; 206,209 images) for testing.
    # Arguments:
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.
    
    # Example:
    train_dataset = l2l.vision.datasets.TieredImagenet(root='./data', mode='train', download=True)
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)
    
    """

    def __init__(self, root, mode='train', transform=None, target_transform=None, download=False):
        super(TieredImagenet, self).__init__()
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        if mode not in ['train', 'validation', 'test']:
            raise ValueError('mode must be train, validation, or test.')
        self.mode = mode
        self._bookkeeping_path = os.path.join(self.root, 'tiered-imagenet-bookkeeping-' + mode + '.pkl')
        google_drive_file_id = '1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07'

        if not self._check_exists() and download:
            self.download(google_drive_file_id, self.root)

        short_mode = 'val' if mode == 'validation' else mode
        tiered_imaganet_path = os.path.join(self.root, 'tiered-imagenet')
        images_path = os.path.join(tiered_imaganet_path, short_mode + '_images_png.pkl')
        with open(images_path, 'rb') as images_file:
            self.images = pickle.load(images_file)
        labels_path = os.path.join(tiered_imaganet_path, short_mode + '_labels.pkl')
        with open(labels_path, 'rb') as labels_file:
            self.labels = pickle.load(labels_file)
            self.labels = self.labels['label_specific']

    def download(self, file_id, destination):
        archive_path = os.path.join(destination, 'tiered_imagenet.tar')
        print('Downloading tiered ImageNet. (12Gb) Please be patient.')
        download_file_from_google_drive(file_id, archive_path)
        archive_file = tarfile.open(archive_path)
        archive_file.extractall(destination)
        os.remove(archive_path)

    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.images[idx]))
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.labels)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           'tiered-imagenet',
                                           'train_images_png.pkl'))

## CUB Meta-Data ##

SPLITS = {
    'train': [
        '190.Red_cockaded_Woodpecker',
        '144.Common_Tern',
        '014.Indigo_Bunting',
        '012.Yellow_headed_Blackbird',
        '059.California_Gull',
        '031.Black_billed_Cuckoo',
        '071.Long_tailed_Jaeger',
        '018.Spotted_Catbird',
        '177.Prothonotary_Warbler',
        '040.Olive_sided_Flycatcher',
        '063.Ivory_Gull',
        '073.Blue_Jay',
        '166.Golden_winged_Warbler',
        '160.Black_throated_Blue_Warbler',
        '016.Painted_Bunting',
        '149.Brown_Thrasher',
        '126.Nelson_Sharp_tailed_Sparrow',
        '090.Red_breasted_Merganser',
        '074.Florida_Jay',
        '058.Pigeon_Guillemot',
        '105.Whip_poor_Will',
        '043.Yellow_bellied_Flycatcher',
        '158.Bay_breasted_Warbler',
        '192.Downy_Woodpecker',
        '129.Song_Sparrow',
        '161.Blue_winged_Warbler',
        '132.White_crowned_Sparrow',
        '146.Forsters_Tern',
        '011.Rusty_Blackbird',
        '070.Green_Violetear',
        '197.Marsh_Wren',
        '041.Scissor_tailed_Flycatcher',
        '100.Brown_Pelican',
        '120.Fox_Sparrow',
        '032.Mangrove_Cuckoo',
        '119.Field_Sparrow',
        '183.Northern_Waterthrush',
        '007.Parakeet_Auklet',
        '053.Western_Grebe',
        '001.Black_footed_Albatross',
        '102.Western_Wood_Pewee',
        '164.Cerulean_Warbler',
        '036.Northern_Flicker',
        '131.Vesper_Sparrow',
        '098.Scott_Oriole',
        '188.Pileated_Woodpecker',
        '139.Scarlet_Tanager',
        '107.Common_Raven',
        '108.White_necked_Raven',
        '184.Louisiana_Waterthrush',
        '099.Ovenbird',
        '171.Myrtle_Warbler',
        '075.Green_Jay',
        '097.Orchard_Oriole',
        '152.Blue_headed_Vireo',
        '173.Orange_crowned_Warbler',
        '095.Baltimore_Oriole',
        '042.Vermilion_Flycatcher',
        '054.Blue_Grosbeak',
        '079.Belted_Kingfisher',
        '006.Least_Auklet',
        '142.Black_Tern',
        '078.Gray_Kingbird',
        '047.American_Goldfinch',
        '050.Eared_Grebe',
        '037.Acadian_Flycatcher',
        '196.House_Wren',
        '083.White_breasted_Kingfisher',
        '062.Herring_Gull',
        '138.Tree_Swallow',
        '060.Glaucous_winged_Gull',
        '182.Yellow_Warbler',
        '027.Shiny_Cowbird',
        '174.Palm_Warbler',
        '157.Yellow_throated_Vireo',
        '117.Clay_colored_Sparrow',
        '175.Pine_Warbler',
        '024.Red_faced_Cormorant',
        '106.Horned_Puffin',
        '151.Black_capped_Vireo',
        '005.Crested_Auklet',
        '185.Bohemian_Waxwing',
        '049.Boat_tailed_Grackle',
        '010.Red_winged_Blackbird',
        '153.Philadelphia_Vireo',
        '017.Cardinal',
        '023.Brandt_Cormorant',
        '115.Brewer_Sparrow',
        '104.American_Pipit',
        '109.American_Redstart',
        '167.Hooded_Warbler',
        '123.Henslow_Sparrow',
        '019.Gray_Catbird',
        '067.Anna_Hummingbird',
        '081.Pied_Kingfisher',
        '077.Tropical_Kingbird',
        '088.Western_Meadowlark',
        '048.European_Goldfinch',
        '141.Artic_Tern',
        '013.Bobolink',
        '029.American_Crow',
        '025.Pelagic_Cormorant',
        '135.Bank_Swallow',
        '056.Pine_Grosbeak',
        '179.Tennessee_Warbler',
        '087.Mallard',
        '195.Carolina_Wren',
        '038.Great_Crested_Flycatcher',
        '092.Nighthawk',
        '187.American_Three_toed_Woodpecker',
        '003.Sooty_Albatross',
        '004.Groove_billed_Ani',
        '156.White_eyed_Vireo',
        '180.Wilson_Warbler',
        '034.Gray_crowned_Rosy_Finch',
        '093.Clark_Nutcracker',
        '110.Geococcyx',
        '154.Red_eyed_Vireo',
        '143.Caspian_Tern',
        '089.Hooded_Merganser',
        '186.Cedar_Waxwing',
        '069.Rufous_Hummingbird',
        '125.Lincoln_Sparrow',
        '026.Bronzed_Cowbird',
        '111.Loggerhead_Shrike',
        '022.Chuck_will_Widow',
        '165.Chestnut_sided_Warbler',
        '021.Eastern_Towhee',
        '191.Red_headed_Woodpecker',
        '086.Pacific_Loon',
        '124.Le_Conte_Sparrow',
        '002.Laysan_Albatross',
        '033.Yellow_billed_Cuckoo',
        '189.Red_bellied_Woodpecker',
        '116.Chipping_Sparrow',
        '130.Tree_Sparrow',
        '114.Black_throated_Sparrow',
        '065.Slaty_backed_Gull',
        '091.Mockingbird',
        '181.Worm_eating_Warbler',
    ],
    'test': [
        '008.Rhinoceros_Auklet',
        '009.Brewer_Blackbird',
        '015.Lazuli_Bunting',
        '020.Yellow_breasted_Chat',
        '028.Brown_Creeper',
        '030.Fish_Crow',
        '035.Purple_Finch',
        '039.Least_Flycatcher',
        '045.Northern_Fulmar',
        '046.Gadwall',
        '082.Ringed_Kingfisher',
        '085.Horned_Lark',
        '094.White_breasted_Nuthatch',
        '101.White_Pelican',
        '103.Sayornis',
        '112.Great_Grey_Shrike',
        '118.House_Sparrow',
        '122.Harris_Sparrow',
        '128.Seaside_Sparrow',
        '133.White_throated_Sparrow',
        '134.Cape_Glossy_Starling',
        '137.Cliff_Swallow',
        '147.Least_Tern',
        '148.Green_tailed_Towhee',
        '163.Cape_May_Warbler',
        '168.Kentucky_Warbler',
        '169.Magnolia_Warbler',
        '170.Mourning_Warbler',
        '193.Bewick_Wren',
        '194.Cactus_Wren',
    ],
    'validation': [
        '044.Frigatebird',
        '051.Horned_Grebe',
        '052.Pied_billed_Grebe',
        '055.Evening_Grosbeak',
        '057.Rose_breasted_Grosbeak',
        '061.Heermann_Gull',
        '064.Ring_billed_Gull',
        '066.Western_Gull',
        '068.Ruby_throated_Hummingbird',
        '072.Pomarine_Jaeger',
        '076.Dark_eyed_Junco',
        '080.Green_Kingfisher',
        '084.Red_legged_Kittiwake',
        '096.Hooded_Oriole',
        '113.Baird_Sparrow',
        '121.Grasshopper_Sparrow',
        '127.Savannah_Sparrow',
        '136.Barn_Swallow',
        '140.Summer_Tanager',
        '145.Elegant_Tern',
        '150.Sage_Thrasher',
        '155.Warbling_Vireo',
        '159.Black_and_white_Warbler',
        '162.Canada_Warbler',
        '172.Nashville_Warbler',
        '176.Prairie_Warbler',
        '178.Swainson_Warbler',
        '198.Rock_Wren',
        '199.Winter_Wren',
        '200.Common_Yellowthroat',
    ]
}

IMAGENET_DUPLICATES = {
    'train': [
        'American_Goldfinch_0062_31921.jpg',
        'Indigo_Bunting_0063_11820.jpg',
        'Blue_Jay_0053_62744.jpg',
        'American_Goldfinch_0131_32911.jpg',
        'Indigo_Bunting_0051_12837.jpg',
        'American_Goldfinch_0012_32338.jpg',
        'Laysan_Albatross_0033_658.jpg',
        'Black_Footed_Albatross_0024_796089.jpg',
        'Indigo_Bunting_0072_14197.jpg',
        'Green_Violetear_0002_795699.jpg',
        'Black_Footed_Albatross_0033_796086.jpg',
        'Black_Footed_Albatross_0086_796062.jpg',
        'Anna_Hummingbird_0034_56614.jpg',
        'American_Goldfinch_0064_32142.jpg',
        'Red_Breasted_Merganser_0068_79203.jpg',
        'Blue_Jay_0033_62024.jpg',
        'Indigo_Bunting_0071_11639.jpg',
        'Red_Breasted_Merganser_0001_79199.jpg',
        'Indigo_Bunting_0060_14495.jpg',
        'Laysan_Albatross_0053_543.jpg',
        'American_Goldfinch_0018_32324.jpg',
        'Red_Breasted_Merganser_0034_79292.jpg',
        'Mallard_0067_77623.jpg',
        'Red_Breasted_Merganser_0083_79562.jpg',
        'Laysan_Albatross_0049_918.jpg',
        'Black_Footed_Albatross_0002_55.jpg',
        'Red_Breasted_Merganser_0012_79425.jpg',
        'Indigo_Bunting_0031_13300.jpg',
        'Blue_Jay_0049_63082.jpg',
        'Indigo_Bunting_0010_13000.jpg',
        'Red_Breasted_Merganser_0004_79232.jpg',
        'Red_Breasted_Merganser_0045_79358.jpg',
        'American_Goldfinch_0116_31943.jpg',
        'Blue_Jay_0068_61543.jpg',
        'Indigo_Bunting_0073_13933.jpg',
    ],
    'validation': [
        'Dark_Eyed_Junco_0057_68650.jpg',
        'Dark_Eyed_Junco_0102_67402.jpg',
        'Ruby_Throated_Hummingbird_0090_57411.jpg',
        'Dark_Eyed_Junco_0031_66785.jpg',
        'Dark_Eyed_Junco_0037_66321.jpg',
        'Dark_Eyed_Junco_0111_66488.jpg',
        'Ruby_Throated_Hummingbird_0040_57982.jpg',
        'Dark_Eyed_Junco_0104_67820.jpg',
    ],
    'test': [],
}

IMAGENET_DUPLICATES['all'] = sum(IMAGENET_DUPLICATES.values(), [])

class CUBirds200(Dataset):

    """
    The dataset consists of 6,033 bird images classified into 200 bird species.
    The train set consists of 140 classes, while the validation and test sets each contain 30.
    This dataset includes 43 images that overlap with the ImageNet dataset.
    # Arguments:
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.
    * **include_imagenet_duplicates** (bool, *optional*, default=False) - Whether to include images that are also present in the ImageNet 2012 dataset.
    
    # Example:
    train_dataset = CUBirds200(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)
    """

    def __init__(
        self,
        root,
        mode='all',
        transform=None,
        target_transform=None,
        download=False,
        include_imagenet_duplicates=False,
        bounding_box_crop=False,
    ):
        root = os.path.expanduser(root)
        self.root = root
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.include_imagenet_duplicates = include_imagenet_duplicates
        self.bounding_box_crop = bounding_box_crop
        self._bookkeeping_path = os.path.join(
            self.root,
            'cubirds200-' + mode + '-bookkeeping.pkl'
        )
        self.DATA_DIR = 'cubirds200'
        self.DATA_FILENAME = 'CUB_200_2011.tgz'
        self.ARCHIVE_ID = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
        # with open('/home/anuj/Desktop/Work/TU_Delft/research/implement/learning_to_meta-learn/data/cub_meta-info.json') as file:
        #     self.cub_data = json.load(file)
        #     file.close()

        if not self._check_exists() and download:
            self.download()

        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, self.DATA_DIR)
        return os.path.exists(data_path)

    def download(self):
        # Download and extract the data
        data_path = os.path.join(self.root, self.DATA_DIR)
        os.makedirs(data_path, exist_ok=True)
        tar_path = os.path.join(data_path, self.DATA_FILENAME)
        print('Downloading CUBirds200 dataset. (1.1Gb)')
        download_file_from_google_drive(self.ARCHIVE_ID, tar_path)
        tar_file = tarfile.open(tar_path)
        tar_file.extractall(data_path)
        tar_file.close()
        os.remove(tar_path)

    def load_data(self, mode='train'):
        classes = sum(SPLITS.values(), []) if mode == 'all' else SPLITS[mode]
        images_path = os.path.join(
            self.root,
            self.DATA_DIR,
            'CUB_200_2011',
            'images',
        )
        duplicates = IMAGENET_DUPLICATES[self.mode]
        self.data = []

        # parse bounding boxes
        if self.bounding_box_crop:
            self.bounding_boxes = {}
            bbox_file = os.path.join(self.root, self.DATA_DIR, 'CUB_200_2011', 'bounding_boxes.txt')
            id2img_file = os.path.join(self.root, self.DATA_DIR, 'CUB_200_2011', 'images.txt')
            with open(bbox_file, 'r') as bbox_fd:
                content = bbox_fd.readlines()
            id2img = {}
            with open(id2img_file, 'r') as id2img_fd:
                for line in id2img_fd.readlines():
                    line = line.replace('\n', '').split(' ')
                    id2img[line[0]] = line[1]
            bbox_content = {}
            for line in content:
                line = line.split(' ')
                x, y, width, height = (
                    int(float(line[1])),
                    int(float(line[2])),
                    int(float(line[3])),
                    int(float(line[4])),
                )
                bbox_content[id2img[line[0]]] = (x, y, x+width, y+height)

        # read images from disk
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(images_path, class_name)
            filenames = os.listdir(class_path)
            for image_file in filenames:
                if self.include_imagenet_duplicates or \
                   image_file not in duplicates:
                    image_path = os.path.join(class_path, image_file)
                    if self.bounding_box_crop:
                        self.bounding_boxes[image_path] = bbox_content[os.path.join(class_name, image_file)]
                    self.data.append((image_path, class_idx))

    def __getitem__(self, i):
        image_path, label = self.data[i]
        image = Image.open(image_path).convert('RGB')
        if self.bounding_box_crop:
            bbox = self.bounding_boxes[image_path]
            image = image.crop(bbox)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        length = len(self.data)
        return length

    # def __init__(
    #     self,
    #     root,
    #     mode='all',
    #     transform=None,
    #     target_transform=None,
    #     download=False,
    #     include_imagenet_duplicates=False,
    #         ):
    #     root = os.path.expanduser(root)
    #     self.root = root
    #     self.mode = mode
    #     self.transform = transform
    #     self.target_transform = target_transform
    #     self.include_imagenet_duplicates = include_imagenet_duplicates
    #     self._bookkeeping_path = os.path.join(
    #         self.root,
    #         'cubirds200-' + mode + '-bookkeeping.pkl'
    #     )
    #     self.DATA_DIR = 'cubirds200'
    #     self.DATA_FILENAME = 'CUB_200_2011.tgz'
    #     self.ARCHIVE_ID = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    #     with open('/home/anuj/Desktop/Work/TU_Delft/research/implement/learning_to_meta-learn/data/cub_meta-info.json') as file:
    #         self.cub_data = json.load(file)
    #         file.close()

    #     if not self._check_exists() and download:
    #         self.download()

    #     self.load_data(mode)

    # def _check_exists(self):
    #     data_path = os.path.join(self.root, self.DATA_DIR)
    #     return os.path.exists(data_path)

    # def download(self):
    #     # Download and extract the data
    #     data_path = os.path.join(self.root, self.DATA_DIR)
    #     os.makedirs(data_path, exist_ok=True)
    #     tar_path = os.path.join(data_path, self.DATA_FILENAME)
    #     print('Downloading CUBirds200 dataset. (1.1Gb)')
    #     download_file_from_google_drive(self.ARCHIVE_ID, tar_path)
    #     tar_file = tarfile.open(tar_path)
    #     tar_file.extractall(data_path)
    #     tar_file.close()
    #     os.remove(tar_path)

    # def load_data(self, mode='train'):
    #     classes = sum(self.cub_data['SPLITS'].values(), []) if mode == 'all' else self.cub_data['SPLITS'][mode]
    #     images_path = os.path.join(
    #         self.root,
    #         self.DATA_DIR,
    #         'CUB_200_2011',
    #         'images',
    #     )
    #     duplicates = self.cub_data['IMAGENET_DUPLICATES'][self.mode]
    #     self.data = []
    #     for class_idx, class_name in enumerate(classes):
    #         class_path = os.path.join(images_path, class_name)
    #         filenames = os.listdir(class_path)
    #         for image_file in filenames:
    #             if self.include_imagenet_duplicates or \
    #                image_file not in duplicates:
    #                 image_path = os.path.join(class_path, image_file)
    #                 self.data.append((image_path, class_idx))

    # def __getitem__(self, i):
    #     image_path, label = self.data[i]
    #     image = default_loader(image_path)
    #     if self.transform is not None:
    #         image = self.transform(image)
    #     if self.target_transform is not None:
    #         label = self.target_transform(label)
    #     return image, label

    # def __len__(self):
    #     length = len(self.data)
    #     return length

class CIFARFS(ImageFolder):

    """
    Consists of 60'000 colour images of sizes 32x32 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
    The classes are sampled from the CIFAR-100 dataset.
    # Arguments: 
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    
    # Example:
    train_dataset = CIFARFS(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    
    """

    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.processed_root = os.path.join(self.root, 'cifarfs', 'processed')
        self.raw_path = os.path.join(self.root, 'cifarfs')

        if not self._check_exists() and download:
            self._download()
        if not self._check_processed():
            self._process_zip()
        mode = 'val' if mode == 'validation' else mode
        self.processed_root = os.path.join(self.processed_root, mode)
        self._bookkeeping_path = os.path.join(self.root, 'cifarfs-' + mode + '-bookkeeping.pkl')
        super(CIFARFS, self).__init__(root=self.processed_root,
                                      transform=self.transform,
                                      target_transform=self.target_transform)

    def _check_exists(self):
        return os.path.exists(self.raw_path)

    def _check_processed(self):
        return os.path.exists(self.processed_root)

    def _download(self):
        # Download the zip, unzip it, and clean up
        print('Downloading CIFARFS to ', self.root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        zip_file = os.path.join(self.root, 'cifarfs.zip')
        download_file_from_google_drive('1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI',
                                        zip_file)
        with zipfile.ZipFile(zip_file, 'r') as zfile:
            zfile.extractall(self.raw_path)
        os.remove(zip_file)

    def _process_zip(self):
        print('Creating CIFARFS splits')
        if not os.path.exists(self.processed_root):
            os.mkdir(self.processed_root)
        split_path = os.path.join(self.raw_path, 'cifar100', 'splits', 'bertinetto')
        train_split_file = os.path.join(split_path, 'train.txt')
        valid_split_file = os.path.join(split_path, 'val.txt')
        test_split_file = os.path.join(split_path, 'test.txt')

        source_dir = os.path.join(self.raw_path, 'cifar100', 'data')
        for fname, dest in [(train_split_file, 'train'),
                            (valid_split_file, 'val'),
                            (test_split_file, 'test')]:
            dest_target = os.path.join(self.processed_root, dest)
            if not os.path.exists(dest_target):
                os.mkdir(dest_target)
            with open(fname) as split:
                for label in split.readlines():
                    source = os.path.join(source_dir, label.strip())
                    target = os.path.join(dest_target, label.strip())
                    shutil.copytree(source, target)


class FGVCAircraft(Dataset):

    """
    The dataset consists of 10,200 images of aircraft (102 classes, each 100 images).
    # Arguments:
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.
    
    # Example:
    train_dataset = l2l.vision.datasets.FGVCAircraft(root='./data', mode='train', download=True)
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)

    """

    def __init__(self, root, mode='all', transform=None, target_transform=None, download=False):
        root = os.path.expanduser(root)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self._bookkeeping_path = os.path.join(self.root, 'fgvc-aircraft-' + mode + '-bookkeeping.pkl')
        self.DATASET_DIR = 'fgvc_aircraft'
        self.DATASET_URL = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
        self.DATA_DIR = os.path.join('fgvc-aircraft-2013b', 'data')
        self.IMAGES_DIR = os.path.join(self.DATA_DIR, 'images')
        self.LABELS_PATH = os.path.join(self.DATA_DIR, 'labels.pkl')
        with open('aircraft_meta-info.json') as file:
            aircraft_data = json.load(file)
            file.close()
        
        if not self._check_exists() and download:
            self.download()

        assert mode in ['train', 'validation', 'test'], \
            'mode should be one of train, validation, test.'
        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, self.DATASET_DIR)
        images_path = os.path.join(data_path, self.IMAGES_DIR)
        labels_path = os.path.join(data_path, self.LABELS_PATH)
        return os.path.exists(data_path) and \
            os.path.exists(images_path) and \
            os.path.exists(labels_path)

    def download(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        data_path = os.path.join(self.root, self.DATASET_DIR)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        tar_path = os.path.join(data_path, os.path.basename(self.DATASET_URL))
        if not os.path.exists(tar_path):
            print('Downloading FGVC Aircraft dataset. (2.75Gb)')
            req = requests.get(self.DATASET_URL)
            with open(tar_path, 'wb') as archive:
                for chunk in req.iter_content(chunk_size=32768):
                    if chunk:
                        archive.write(chunk)
        with tarfile.open(tar_path) as tar_file:
            tar_file.extractall(data_path)
        family_names = ['images_family_train.txt',
                        'images_family_val.txt',
                        'images_family_test.txt']
        images_labels = []
        for family in family_names:
            with open(os.path.join(data_path, self.DATA_DIR, family_names[0]), 'r') as family_file:
                for line in family_file.readlines():
                    image, label = line.split(' ', 1)
                    images_labels.append((image.strip(), label.strip()))
        labels_path = os.path.join(data_path, self.LABELS_PATH)
        with open(labels_path, 'wb') as labels_file:
            pickle.dump(images_labels, labels_file)
        os.remove(tar_path)

    def load_data(self, mode='train'):
        data_path = os.path.join(self.root, self.DATASET_DIR)
        labels_path = os.path.join(data_path, self.LABELS_PATH)
        with open(labels_path, 'rb') as labels_file:
            image_labels = pickle.load(labels_file)

        data = []
        mode = 'valid' if mode == 'validation' else mode
        split = self.aircraft_data['SPLITS'][mode]
        for image, label in image_labels:
            if label in split:
                image = os.path.join(data_path, self.IMAGES_DIR, image + '.jpg')
                label = split.index(label)
                data.append((image, label))
        self.data = data

    def __getitem__(self, i):
        image, label = self.data[i]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data)


class FGVCFungi(Dataset):

    """
    The dataset consists of 1,394 classes and 89,760 images of fungi.
    # Arguments:
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.
    
    # Example:
    train_dataset = l2l.vision.datasets.FGVCFungi(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)

    """

    def __init__(self, root, mode='all', transform=None, target_transform=None, download=False):
        root = os.path.expanduser(root)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self._bookkeeping_path = os.path.join(self.root, 'fgvc-fungi-' + mode + '-bookkeeping.pkl')
        self.DATA_DIR = 'fgvc_fungi'
        self.DATA_URL = 'https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz'
        self.ANNOTATIONS_URL = 'https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz'
        with open('fungi_meta-info.json') as file:
            fungi_data = json.load(file)
            file.close()

        if not self._check_exists() and download:
            self.download()

        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, self.DATA_DIR, 'images')
        annotations_path = os.path.join(self.root, self.DATA_DIR, 'train.json')
        return os.path.exists(data_path)

    def download(self):
        data_path = os.path.join(self.root, self.DATA_DIR)
        os.makedirs(data_path, exist_ok=True)
        data_tar_path = os.path.join(data_path, os.path.basename(self.DATA_URL))
        annotations_tar_path = os.path.join(data_path, os.path.basename(self.ANNOTATIONS_URL))

        # Download data
        print('Downloading FGVC Fungi dataset (12.9Gb)')
        download_file(self.DATA_URL, data_tar_path, size=12_900_000_000)
        download_file(self.ANNOTATIONS_URL, annotations_tar_path)

        # Extract data
        tar_file = tarfile.open(data_tar_path)
        tar_file.extractall(data_path)
        tar_file.close()
        os.remove(data_tar_path)

        # Extract annotations
        tar_file = tarfile.open(annotations_tar_path)
        tar_file.extractall(data_path)
        tar_file.close()
        os.remove(annotations_tar_path)

    def load_data(self, mode='train'):
        if not os.path.exists(self._bookkeeping_path):
            # Load annotations
            data_path = os.path.join(self.root, self.DATA_DIR)
            train_path = os.path.join(data_path, 'train.json')
            with open(train_path, 'r') as f_train:
                train_annotations = json.load(f_train)
            valid_path = os.path.join(data_path, 'val.json')
            with open(valid_path, 'r') as f_valid:
                valid_annotations = json.load(f_valid)
            split_classes = sum(self.fungi_data['SPLITS'].values(), []) if mode == 'all' else self.fungi_data['SPLITS'][mode]
            split_classes = [int(cls[:4]) for cls in split_classes]

            # Create bookkeeping
            labels_to_indices = defaultdict(list)
            indices_to_labels = defaultdict(int)
            data_map = []

            # Process
            all_images = train_annotations['images'] + valid_annotations['images']
            all_annotations = train_annotations['annotations'] \
                + valid_annotations['annotations']
            counter = 0
            for image, annotation in zip(all_images, all_annotations):
                assert image['id'] == annotation['image_id']
                img_cat = annotation['category_id']
                if img_cat in split_classes:
                    img_path = os.path.join(data_path, image['file_name'])
                    label = split_classes.index(img_cat)
                    data_map.append((img_path, label))
                    labels_to_indices[label].append(counter)
                    indices_to_labels[counter] = label
                    counter += 1

            # Cache to disk
            bookkeeping = {
                'labels_to_indices': labels_to_indices,
                'indices_to_labels': indices_to_labels,
                'labels': list(labels_to_indices.keys()),
                'data_map': data_map,
            }
            with open(self._bookkeeping_path, 'wb') as f:
                pickle.dump(bookkeeping, f, protocol=-1)
        else:
            # Load bookkeeping
            with open(self._bookkeeping_path, 'rb') as f:
                bookkeeping = pickle.load(f)

        self._bookkeeping = bookkeeping
        self.labels_to_indices = bookkeeping['labels_to_indices']
        self.indices_to_labels = bookkeeping['indices_to_labels']
        self.labels = bookkeeping['labels']
        self.data_map = bookkeeping['data_map']
        self.length = len(self.indices_to_labels)

    def __getitem__(self, i):
        image, label = self.data_map[i]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return self.length


class VGGFlower102(Dataset):

    """
    The dataset consists of 102 classes of flowers, with each class consisting of 40 to 258 images.
    # Arguments:
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.
    
    # Example:
    train_dataset = l2l.vision.datasets.VGGFlower102(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)
    
    """

    def __init__(self, root, mode='all', transform=None, target_transform=None, download=False):
        root = os.path.expanduser(root)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self._bookkeeping_path = os.path.join(self.root, 'vgg-flower102-' + mode + '-bookkeeping.pkl')
        self.DATA_DIR = 'vgg_flower102'
        self.IMAGES_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
        self.LABELS_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
        self.IMAGES_DIR = 'jpg'
        self.LABELS_PATH = 'imagelabels.mat'

        self.SPLITS = {
    'train': [90, 38, 80, 30, 29, 12, 43, 27, 4, 64, 31, 99, 8, 67, 95, 77,
              78, 61, 88, 74, 55, 32, 21, 13, 79, 70, 51, 69, 14, 60, 11, 39,
              63, 37, 36, 28, 48, 7, 93, 2, 18, 24, 6, 3, 44, 76, 75, 72, 52,
              84, 73, 34, 54, 66, 59, 50, 91, 68, 100, 71, 81, 101, 92, 22,
              33, 87, 1, 49, 20, 25, 58],
    'validation': [10, 16, 17, 23, 26, 47, 53, 56, 57, 62, 82, 83, 86, 97, 102],
    'test': [5, 9, 15, 19, 35, 40, 41, 42, 45, 46, 65, 85, 89, 94, 96, 98],
    'all': list(range(1, 103)),
}

        if not self._check_exists() and download:
            self.download()

        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, self.DATA_DIR)
        return os.path.exists(data_path)

    def download(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        data_path = os.path.join(self.root, self.DATA_DIR)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        tar_path = os.path.join(data_path, os.path.basename(self.IMAGES_URL))
        print('Downloading VGG Flower102 dataset (330Mb)')
        download_file(self.IMAGES_URL, tar_path)
        tar_file = tarfile.open(tar_path)
        tar_file.extractall(data_path)
        tar_file.close()
        os.remove(tar_path)

        label_path = os.path.join(data_path, os.path.basename(self.LABELS_URL))
        req = requests.get(self.LABELS_URL)
        with open(label_path, 'wb') as label_file:
            label_file.write(req.content)

    def load_data(self, mode='train'):
        data_path = os.path.join(self.root, self.DATA_DIR)
        images_path = os.path.join(data_path, self.IMAGES_DIR)
        labels_path = os.path.join(data_path, self.LABELS_PATH)
        labels_mat = scipy.io.loadmat(labels_path)
        image_labels = []
        split = self.SPLITS[mode]
        for idx, label in enumerate(labels_mat['labels'][0], start=1):
            if label in split:
                image = str(idx).zfill(5)
                image = f'image_{image}.jpg'
                image = os.path.join(images_path, image)
                label = split.index(label)
                image_labels.append((image, label))
        self.data = image_labels

    def __getitem__(self, i):
        image, label = self.data[i]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data)
