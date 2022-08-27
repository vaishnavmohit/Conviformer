# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import random
import pandas as pd
from PIL import Image
from sklearn import preprocessing
import numpy as np

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, DatasetFolder, default_loader

from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    return filename.lower().endswith(extensions)

def make_subsampled_dataset(
        directory, class_to_idx, extensions=None,is_valid_file=None, sampling_ratio=1., nb_classes=None):

    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for i, target_class in enumerate(sorted(class_to_idx.keys())):
        if nb_classes is not None and i>=nb_classes:
            break
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        num_imgs = int(len(os.listdir(target_dir))*sampling_ratio)
        imgs=0
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if imgs==num_imgs :
                    break
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    imgs+=1
    return instances

class NABirds(Dataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = '/gpfs/scratch/mvaishn1/datasets/nabirds/images'

    def __init__(self, root='/gpfs/scratch/mvaishn1/datasets/', train=True, transform=None):
        dataset_path = os.path.join(root, 'nabirds')
        self.root = root
        self.loader = default_loader
        self.train = train
        self.transform = transform

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}

def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names

def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents

def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic

class INAT(Dataset):
    def __init__(self, root, ann_file, transforms):
        self.transforms = transforms
        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print( '\t' + str(len(self.imgs)) + ' images')
        print( '\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.loader = default_loader

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        # im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        # tax_ids = self.classes_taxonomic[species_id]

        img = self.transforms(img)

        return img, species_id

    def __len__(self):
        return len(self.imgs)


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class SubsampledDatasetFolder(DatasetFolder):

    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, sampling_ratio=1., nb_classes=None):

        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_subsampled_dataset(self.root, class_to_idx, extensions, is_valid_file, sampling_ratio=sampling_ratio, nb_classes=nb_classes)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        

class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir:             
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]

class GetData_raw(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
        x = x.crop((0,0,540,960))
        # import pdb; pdb.set_trace()
    
        if "train" in self.dir:             
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]

class GetDataTest(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform, Ids):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels   
        self.ids = Ids      
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
              
        return self.transform(x), self.fnames[index], self.ids[index]


# https://www.kaggle.com/salaheddinelahmadi/exploration-preprocessing-baseline-cnn
class GetDataHier(Dataset):
    def __init__(self, Dir, FNames, Labels, Family, Genus, Species, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels 
        self.family = Family
        self.genus = Genus
        self.species = Species        
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir:             
            return self.transform(x), self.labels[index], self.family[index], self.genus[index], self.species[index]
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]

class GetDataTriplet(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform, item):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels      
        self.item = item
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):  
          
        anchor_img = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir or "" in self.dir:   
            anchor_label = self.labels[index]
            positive_list = self.item[self.item!=index][self.labels[self.item!=index]==anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = Image.open(os.path.join(self.dir, self.fnames[positive_item]))
            pos_label = self.labels[positive_item]
            
            negative_list = self.item[self.item!=index][self.labels[self.item!=index]!=anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = Image.open(os.path.join(self.dir, self.fnames[negative_item]))
            neg_label = self.labels[negative_item]

            return self.transform(anchor_img), self.transform(positive_img), self.transform(negative_img), anchor_label, pos_label, neg_label

        elif "test" in self.dir:            
            return self.transform(anchor_img), self.fnames[index]

class GetDataGenus(Dataset):
    def __init__(self, Dir, FNames, Labels, Genus, Transform, item):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels   
        self.genus = Genus    
        self.item = item
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        anchor_img = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir:   
            anchor_label = self.labels[index]
            genus_label = self.genus[index] 
            positive_list = self.item[self.item!=index][self.genus[self.item!=index]==genus_label]

            positive_item = random.choice(positive_list)
            positive_img = Image.open(os.path.join(self.dir, self.fnames[positive_item]))
            anchor_label_p = self.labels[positive_item]
            genus_label_p = self.genus[positive_item]
            
            negative_list = self.item[self.item!=index][self.genus[self.item!=index]!=genus_label]
            negative_item = random.choice(negative_list)
            negative_img = Image.open(os.path.join(self.dir, self.fnames[negative_item]))
            anchor_label_n = self.labels[negative_item]
            genus_label_n = self.genus[negative_item]

            return self.transform(anchor_img), self.transform(positive_img), self.transform(negative_img), anchor_label, anchor_label_p, anchor_label_n, genus_label, genus_label_p, genus_label_n

        elif "test" in self.dir:            
            return self.transform(anchor_img), self.fnames[index]

class GetDataSpecies(Dataset):
    def __init__(self, Dir, FNames, Labels, Genus, Transform, item):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels   
        self.genus = Genus    
        self.item = item
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        anchor_img = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir:   
            anchor_label = self.labels[index]
            genus_label = self.genus[index] 
            positive_list = self.item[self.item!=index][self.labels[self.item!=index]==anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = Image.open(os.path.join(self.dir, self.fnames[positive_item]))
            anchor_label_p = self.labels[positive_item]
            genus_label_p = self.genus[positive_item]
            
            negative_list = self.item[self.item!=index][self.labels[self.item!=index]!=anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = Image.open(os.path.join(self.dir, self.fnames[negative_item]))
            anchor_label_n = self.labels[negative_item]
            genus_label_n = self.genus[negative_item]

            return self.transform(anchor_img), self.transform(positive_img), self.transform(negative_img), anchor_label, anchor_label_p, anchor_label_n, genus_label, genus_label_p, genus_label_n

        elif "test" in self.dir:            
            return self.transform(anchor_img), self.fnames[index]

class GetDataHierSpecies(Dataset):
    def __init__(self, Dir, FNames, Labels, Genus, Transform, item, family):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels   
        self.genus = Genus   
        self.family = family    
        self.item = item
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        anchor_img = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir:   
            anchor_label = self.labels[index]
            genus_label = self.genus[index] 
            family_label = self.family[index] 
            # positive_list = self.item[self.item!=index][self.labels[self.item!=index]==anchor_label]
            fam_list = self.item[self.item!=index][self.family[self.item!=index]==family_label]
            gen_list = fam_list[self.genus[fam_list]==genus_label]
            positive_list = gen_list[self.labels[gen_list]==anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = Image.open(os.path.join(self.dir, self.fnames[positive_item]))
            anchor_label_p = self.labels[positive_item]
            genus_label_p = self.genus[positive_item]
            family_label_p = self.family[positive_item] 
            
            # negative_list = self.item[self.item!=index][self.labels[self.item!=index]!=anchor_label]
            # negative_list = self.item[self.item!=index][self.family[self.item!=index]!=family_label][self.genus[self.item!=index]!=genus_label][self.labels[self.item!=index]!=anchor_label]
            n_fam_list = self.item[self.item!=index][self.family[self.item!=index]!=family_label]
            n_gen_list = n_fam_list[self.genus[n_fam_list]!=genus_label]
            negative_list = n_gen_list[self.labels[n_gen_list]!=anchor_label]

            negative_item = random.choice(negative_list)
            negative_img = Image.open(os.path.join(self.dir, self.fnames[negative_item]))
            anchor_label_n = self.labels[negative_item]
            genus_label_n = self.genus[negative_item]
            family_label_n = self.family[negative_item] 

            return self.transform(anchor_img), self.transform(positive_img), self.transform(negative_img), anchor_label, anchor_label_p, anchor_label_n, genus_label, genus_label_p, genus_label_n, family_label, family_label_p, family_label_n

        elif "test" in self.dir:            
            return self.transform(anchor_img), self.fnames[index]

class GetDataHierGenus(Dataset):
    def __init__(self, Dir, FNames, Labels, Genus, Transform, item, family):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels   
        self.genus = Genus   
        self.family = family    
        self.item = item
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        anchor_img = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir:   
            anchor_label = self.labels[index]
            genus_label = self.genus[index] 
            family_label = self.family[index] 
            positive_list = self.item[self.item!=index][self.genus[self.item!=index]==genus_label]

            positive_item = random.choice(positive_list)
            positive_img = Image.open(os.path.join(self.dir, self.fnames[positive_item]))
            anchor_label_p = self.labels[positive_item]
            genus_label_p = self.genus[positive_item]
            family_label_p = self.family[positive_item] 
            
            negative_list = self.item[self.item!=index][self.genus[self.item!=index]!=genus_label]
            negative_item = random.choice(negative_list)
            negative_img = Image.open(os.path.join(self.dir, self.fnames[negative_item]))
            anchor_label_n = self.labels[negative_item]
            genus_label_n = self.genus[negative_item]
            family_label_n = self.family[negative_item] 

            return self.transform(anchor_img), self.transform(positive_img), self.transform(negative_img), anchor_label, anchor_label_p, anchor_label_n, genus_label, genus_label_p, genus_label_n, family_label, family_label_p, family_label_n

        elif "test" in self.dir:            
            return self.transform(anchor_img), self.fnames[index]

class ImageNetDataset(SubsampledDatasetFolder):
    def __init__(self, root, loader=default_loader, is_valid_file=None,  **kwargs):
        super(ImageNetDataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                              is_valid_file=is_valid_file, **kwargs)
        self.imgs = self.samples


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        args.data_path = "/datasets01/cifar-pytorch/11222017/"
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    
    if args.data_set == 'CIFAR100':
        args.data_path = "/datasets01/cifar100/022818/data/"
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
   
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = ImageNetDataset(root, transform=transform,
                                  sampling_ratio= (args.sampling_ratio if is_train else 1.), nb_classes=args.nb_classes)
        nb_classes = args.nb_classes if args.nb_classes is not None else 1000
    
    elif args.data_set == 'NABIRD':
        train_full_data = NABirds(train=True, transform=transform)
        # import pdb; pdb.set_trace()
        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), \
            test_size=.1, stratify=train_full_data.data['target'])
        print(len(val_idx))
        print(len(train_idx))
        train_data = Subset(train_full_data, train_idx) # train_full_data # 
        val_data = Subset(train_full_data, val_idx)

        test_data = NABirds(train=False, transform=transform)
        nb_classes = 555
        if is_train=='train':
            dataset = train_data
        elif is_train=='val':
            dataset = val_data
        elif is_train=='test':
            dataset = test_data
        else:
            print('Wrong Dataset')
            exit()

    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    elif args.data_set == 'INAT19':
        args.data_path = "/gpfs/scratch/mvaishn1/datasets/inaturalist-2019-fgvc6/"
        train_file = os.path.join(args.data_path, 'train2019.json')
        val_file = os.path.join(args.data_path, 'val2019.json')
        test_file = os.path.join(args.data_path, 'test2019.json')
        with open(train_file) as data_file:
            train_data = json.load(data_file)
        with open(val_file) as data_file:
            val_data = json.load(data_file)
        with open(test_file) as data_file:
            test_data = json.load(data_file)
        
        root = os.path.join(args.data_path)
        # https://www.kaggle.com/code/interneuron/previous-benchmark-in-a-kernel-v0-0-0
        train_data = INAT(root=root, ann_file=train_file, transforms=transform)
        val_data = INAT(root=root, ann_file=val_file, transforms=transform)

        if is_train=='train':
            dataset = train_data
        elif is_train=='val':
            dataset = val_data
        elif is_train=='test':
            dataset = val_data
        else:
            print('Wrong Dataset')
            exit()
        nb_classes = 1010 

    
    elif args.data_set == 'Herbarium':
        # TRAIN_DIR = "/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8/train/"
        TRAIN_DIR = "/users/mvaishn1/scratch/datasets/Herbarium_2021_FGVC8/train/"
        with open(TRAIN_DIR + 'metadata.json', "r", encoding="ISO-8859-1") as file:
            train = json.load(file)
        train_img = pd.DataFrame(train['images'])
        train_ann = pd.DataFrame(train['annotations']).drop(columns='image_id')
        df = train_img.merge(train_ann, on='id')
        # print(len(df))
        nb_classes = len(df['category_id'].value_counts())
        # print(nb_classes)
        X_Train, Y_Train = df['file_name'].values, df['category_id'].values
        train_full_data = GetData(TRAIN_DIR, X_Train, Y_Train, transform)

        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), test_size=.1, stratify=Y_Train)
        print(len(val_idx))
        print(len(train_idx))
        train_data = Subset(train_full_data, train_idx) # train_full_data # 
        val_data = Subset(train_full_data, val_idx)

        # TEST DATA:
        # TEST_DIR = "/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8/test/"
        # TEST_DIR = "/users/mvaishn1/scratch/datasets/Herbarium_2021_FGVC8/test/"
        # with open(TEST_DIR + 'metadata.json', "r", encoding="ISO-8859-1") as file:
        #     test = json.load(file)
        # df_test = pd.DataFrame(test['images'])
        # test_ann = pd.DataFrame(test['annotations']).drop(columns='image_id')
        # df_test = train_img.merge(test_ann, on='id')
        # print(len(df_test))
        # NUM_CL_test = len(df_test['category_id'].value_counts())
        # print(NUM_CL_test)
        # X_Test = df['file_name'].values
        # Y_Test = None
        # test_data =GetData(TEST_DIR, X_Test, Y_Test, transform)

        if is_train=='train':
            dataset = train_data
        elif is_train=='val':
            dataset = val_data
        elif is_train=='test':
            dataset = val_data
        else:
            print('Wrong Dataset')
            exit()
    
    elif args.data_set == 'Herbarium22':
        Base_DIR = "/users/mvaishn1/scratch/datasets/herbarium-2022-fgvc9_resize/"
        train_dir = Base_DIR + "train_images/"
        with open(Base_DIR + 'train_metadata.json', "r", encoding="ISO-8859-1") as file:
            train_meta = json.load(file)
        image_ids = [image["image_id"] for image in train_meta["images"]]
        image_dirs = [train_dir + image['file_name'] for image in train_meta["images"]]
        category_ids = [annotation['category_id'] for annotation in train_meta['annotations']]
        genus_ids = [annotation['genus_id'] for annotation in train_meta['annotations']]

        df = pd.DataFrame({
                            "image_id" : image_ids,
                            "image_dir" : image_dirs,
                            "category_id" : category_ids,
                            "genus" : genus_ids})

        nb_classes = len(df['category_id'].value_counts())
        le = preprocessing.LabelEncoder()
        le.fit(df['category_id'].values)

        # from pathlib import Path  
        # filepath = Path('/users/mvaishn1/data/data/mvaishn1/fossil/submission_herb22/onehot.csv')  
        # filepath.parent.mkdir(parents=True, exist_ok=True)  
        # import pdb; pdb.set_trace()
        # df.to_csv(filepath)
        # df2 = pd.DataFrame({'labels': df['category_id'].values, 'inverse': le.transform(df['category_id'].values)})
        # filepath = Path('/users/mvaishn1/data/data/mvaishn1/fossil/submission_herb22/labels.csv')
        # df2.to_csv(filepath)
        
        print(nb_classes)
        X_Train, Y_Train = df['image_dir'].values, le.transform(df['category_id'].values)
        # le.inverse_transform([0, 0, 1, 2]) --> inverse transform
        if args.triplet:
            train_full_data = GetDataTriplet(train_dir, X_Train, Y_Train, transform, df.index.values)
        else:
            train_full_data = GetData(train_dir, X_Train, Y_Train, transform)

        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), test_size=.12, stratify=Y_Train)
        print(len(val_idx))
        print(len(train_idx))
        # if 'hier' not in args.model and args.loss_type == 'ce':
        #     train_data = Subset(train_full_data, train_idx)  #train_full_data #only for ce evaluation for more epochs
        #     _, val_idx = train_test_split(list(range(len(train_full_data))), test_size=.80)
        # else:
        train_data = Subset(train_full_data, train_idx) 
        val_data = Subset(train_full_data, val_idx)

        # TEST DATA:
        Base_DIR = "/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize/"
        test_dir = Base_DIR + "test_images/"
        with open(Base_DIR + 'test_metadata.json', "r", encoding="ISO-8859-1") as file:
            test_meta = json.load(file)
        test_ids = [image['image_id'] for image in test_meta]
        test_dirs = [test_dir + image['file_name'] for image in test_meta]
        df_test = pd.DataFrame({
                                "test_id" : test_ids,
                                "test_dir" : test_dirs
                            })
        # print(len(df_test))
        NUM_CL_test = len(df_test['test_dir'].value_counts())
        print(NUM_CL_test)
        X_Test = df_test['test_dir'].values
        Y_Test = None
        test_data =GetDataTest(test_dir, X_Test, Y_Test, transform, df_test['test_id'].values)

        if is_train=='train':
            dataset = train_data
        elif is_train=='val':
            dataset = val_data
        elif is_train=='test':
            dataset = test_data
        else:
            print('Wrong Dataset')
            exit()

    elif args.data_set == 'Herbarium22_raw':
        Base_DIR = "/users/mvaishn1/scratch/datasets/herbarium-2022-fgvc9_resize/"
        train_dir = Base_DIR + "train_images/"
        with open(Base_DIR + 'train_metadata.json', "r", encoding="ISO-8859-1") as file:
            train_meta = json.load(file)
        image_ids = [image["image_id"] for image in train_meta["images"]]
        image_dirs = [train_dir + image['file_name'] for image in train_meta["images"]]
        category_ids = [annotation['category_id'] for annotation in train_meta['annotations']]
        genus_ids = [annotation['genus_id'] for annotation in train_meta['annotations']]

        df = pd.DataFrame({
                            "image_id" : image_ids,
                            "image_dir" : image_dirs,
                            "category_id" : category_ids,
                            "genus" : genus_ids})

        nb_classes = len(df['category_id'].value_counts())
        le = preprocessing.LabelEncoder()
        le.fit(df['category_id'].values)
        
        print(nb_classes)
        X_Train, Y_Train = df['image_dir'].values, le.transform(df['category_id'].values)
        
        train_full_data = GetData_raw(train_dir, X_Train, Y_Train, transform)

        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), test_size=.12, stratify=Y_Train)
        print(len(val_idx))
        print(len(train_idx))

        train_data = Subset(train_full_data, train_idx) 
        val_data = Subset(train_full_data, val_idx)

        # TEST DATA:
        Base_DIR = "/users/mvaishn1/scratch/datasets/herbarium-2022-fgvc9/"
        test_dir = Base_DIR + "test_images/"
        with open(Base_DIR + 'test_metadata.json', "r", encoding="ISO-8859-1") as file:
            test_meta = json.load(file)
        test_ids = [image['image_id'] for image in test_meta]
        test_dirs = [test_dir + image['file_name'] for image in test_meta]
        df_test = pd.DataFrame({
                                "test_id" : test_ids,
                                "test_dir" : test_dirs
                            })
        # print(len(df_test))
        NUM_CL_test = len(df_test['test_dir'].value_counts())
        print(NUM_CL_test)
        X_Test = df_test['test_dir'].values
        Y_Test = None
        test_data =GetDataTest(test_dir, X_Test, Y_Test, transform, df_test['test_id'].values)

        if is_train=='train':
            dataset = train_data
        elif is_train=='val':
            dataset = val_data
        elif is_train=='test':
            dataset = test_data
        else:
            print('Wrong Dataset')
            exit()
    

    elif args.data_set == 'Herbarium22hier':
        Base_DIR = "/users/mvaishn1/scratch/datasets/herbarium-2022-fgvc9_resize/"
        train_dir = Base_DIR + "train_images/"
        with open(Base_DIR + 'train_metadata.json', "r", encoding="ISO-8859-1") as file:
            train_meta = json.load(file)
        ids = []
        categories = []
        paths = []

        for annotation, image in zip(train_meta["annotations"], train_meta["images"]):
            ids.append(image["image_id"]) #Read above print metadata samples from each key
            categories.append(annotation["category_id"])
            paths.append(train_dir + image["file_name"])
        
        df_meta = pd.DataFrame({"id": ids, "category": categories, "image_dir": paths})

        d_categories = {category["category_id"]: category["scientificName"] for category in train_meta["categories"]}
        d_families = {category["category_id"]: category["family"] for category in train_meta["categories"]}
        d_genus = {category["category_id"]: category["genus"] for category in train_meta["categories"]}
        d_species = {category["category_id"]: category["species"] for category in train_meta["categories"]}

        df_meta["category_id"] = df_meta["category"].map(d_categories) # 15501
        df_meta["family_name"] = df_meta["category"].map(d_families) # 272
        df_meta["genus_name"] = df_meta["category"].map(d_genus) # 2564
        df_meta["species_name"] = df_meta["category"].map(d_species) # 6932
        df = df_meta

        nb_classes = len(df['category_id'].value_counts())
        nb_family = len(df['family_name'].value_counts())
        nb_genus = len(df['genus_name'].value_counts())
        nb_species = len(df['species_name'].value_counts())
        # category encoder
        le = preprocessing.LabelEncoder()
        le.fit(df['category_id'].values)
        # family encoder 
        lef = preprocessing.LabelEncoder()
        lef.fit(df['family_name'].values)
        # genus encoder
        leg = preprocessing.LabelEncoder()
        leg.fit(df['genus_name'].values)
        # species encoder
        les = preprocessing.LabelEncoder()
        les.fit(df['species_name'].values)

        print(nb_classes)
        
        Y_family = lef.transform(df['family_name'].values)
        Y_genus = leg.transform(df['genus_name'].values)
        Y_species = les.transform(df['species_name'].values)
        X_Train, Y_Train = df['image_dir'].values, le.transform(df['category_id'].values)
        train_full_data = GetDataHier(train_dir, X_Train, Y_Train, Y_family, \
            Y_genus, Y_species,  transform)

        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), test_size=.12, stratify=Y_Train)
        print(len(val_idx))
        print(len(train_idx))
        train_data = Subset(train_full_data, train_idx) # train_full_data #
        val_data = Subset(train_full_data, val_idx)

        # TEST DATA:
        Base_DIR = "/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize/"
        test_dir = Base_DIR + "test_images/"
        with open(Base_DIR + 'test_metadata.json', "r", encoding="ISO-8859-1") as file:
            test_meta = json.load(file)
        test_ids = [image['image_id'] for image in test_meta]
        test_dirs = [test_dir + image['file_name'] for image in test_meta]
        df_test = pd.DataFrame({
                                "test_id" : test_ids,
                                "test_dir" : test_dirs
                            })
        # print(len(df_test))
        NUM_CL_test = len(df_test['test_dir'].value_counts())
        print(NUM_CL_test)
        X_Test = df_test['test_dir'].values
        Y_Test = None
        test_data = GetDataTest(test_dir, X_Test, Y_Test, transform, df_test['test_id'].values)

        if is_train=='train':
            dataset = train_data
        elif is_train=='val':
            dataset = val_data
        elif is_train=='test':
            dataset = test_data
        else:
            print('Wrong Dataset')
            exit()

        return dataset, nb_classes 
    
    elif args.data_set == 'Herbarium22genus':
        Base_DIR = "/users/mvaishn1/scratch/datasets/herbarium-2022-fgvc9_resize/"
        train_dir = Base_DIR + "train_images/"
        with open(Base_DIR + 'train_metadata.json', "r", encoding="ISO-8859-1") as file:
            train_meta = json.load(file)
        ids = []
        categories = []
        paths = []

        for annotation, image in zip(train_meta["annotations"], train_meta["images"]):
            ids.append(image["image_id"]) #Read above print metadata samples from each key
            categories.append(annotation["category_id"])
            paths.append(train_dir + image["file_name"])
        
        df_meta = pd.DataFrame({"id": ids, "category": categories, "image_dir": paths})

        d_categories = {category["category_id"]: category["scientificName"] for category in train_meta["categories"]}
        d_families = {category["category_id"]: category["family"] for category in train_meta["categories"]}
        d_genus = {category["category_id"]: category["genus"] for category in train_meta["categories"]}
        d_species = {category["category_id"]: category["species"] for category in train_meta["categories"]}

        df_meta["category_id"] = df_meta["category"].map(d_categories) # 15501
        df_meta["family_name"] = df_meta["category"].map(d_families) # 272
        df_meta["genus_name"] = df_meta["category"].map(d_genus) # 2564
        df_meta["species_name"] = df_meta["category"].map(d_species) # 6932
        df = df_meta

        nb_classes = len(df['category_id'].value_counts())
        nb_family = len(df['family_name'].value_counts())
        nb_genus = len(df['genus_name'].value_counts())
        nb_species = len(df['species_name'].value_counts())
        # category encoder
        le = preprocessing.LabelEncoder()
        le.fit(df['category_id'].values)
        # family encoder 
        lef = preprocessing.LabelEncoder()
        lef.fit(df['family_name'].values)
        # genus encoder
        leg = preprocessing.LabelEncoder()
        leg.fit(df['genus_name'].values)
        # species encoder
        les = preprocessing.LabelEncoder()
        les.fit(df['species_name'].values)

        print(nb_classes)
        
        Y_family = lef.transform(df['family_name'].values)
        Y_genus = leg.transform(df['genus_name'].values)
        Y_species = les.transform(df['species_name'].values)
        X_Train, Y_Train = df['image_dir'].values, le.transform(df['category_id'].values)
        train_full_data = GetDataGenus(train_dir, X_Train, Y_Train, Y_genus, transform, df.index.values)

        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), test_size=.12, stratify=Y_Train)
        print(len(val_idx))
        print(len(train_idx))
        train_data = Subset(train_full_data, train_idx) # train_full_data #
        val_data = Subset(train_full_data, val_idx)

        # TEST DATA:
        test_dir = Base_DIR + "test_images/"
        with open(Base_DIR + 'test_metadata.json', "r", encoding="ISO-8859-1") as file:
            test_meta = json.load(file)
        test_ids = [image['image_id'] for image in test_meta]
        test_dirs = [test_dir + image['file_name'] for image in test_meta]
        df_test = pd.DataFrame({
                                "test_id" : test_ids,
                                "test_dir" : test_dirs
                            })
        # print(len(df_test))
        NUM_CL_test = len(df_test['test_dir'].value_counts())
        print(NUM_CL_test)
        X_Test = df_test['test_dir'].values
        Y_Test = None
        test_data =GetData(test_dir, X_Test, Y_Test, transform)

        if is_train=='train':
            dataset = train_data
        elif is_train=='val':
            dataset = val_data
        elif is_train=='test':
            dataset = val_data
        else:
            print('Wrong Dataset')
            exit()
    
    elif args.data_set == 'Herbarium22hiergenus':
        Base_DIR = "/users/mvaishn1/scratch/datasets/herbarium-2022-fgvc9_resize/"
        train_dir = Base_DIR + "train_images/"
        with open(Base_DIR + 'train_metadata.json', "r", encoding="ISO-8859-1") as file:
            train_meta = json.load(file)
        ids = []
        categories = []
        paths = []

        for annotation, image in zip(train_meta["annotations"], train_meta["images"]):
            ids.append(image["image_id"]) #Read above print metadata samples from each key
            categories.append(annotation["category_id"])
            paths.append(train_dir + image["file_name"])
        
        df_meta = pd.DataFrame({"id": ids, "category": categories, "image_dir": paths})

        d_categories = {category["category_id"]: category["scientificName"] for category in train_meta["categories"]}
        d_families = {category["category_id"]: category["family"] for category in train_meta["categories"]}
        d_genus = {category["category_id"]: category["genus"] for category in train_meta["categories"]}
        d_species = {category["category_id"]: category["species"] for category in train_meta["categories"]}

        df_meta["category_id"] = df_meta["category"].map(d_categories) # 15501
        df_meta["family_name"] = df_meta["category"].map(d_families) # 272
        df_meta["genus_name"] = df_meta["category"].map(d_genus) # 2564
        df_meta["species_name"] = df_meta["category"].map(d_species) # 6932
        df = df_meta

        nb_classes = len(df['category_id'].value_counts())
        nb_family = len(df['family_name'].value_counts())
        nb_genus = len(df['genus_name'].value_counts())
        nb_species = len(df['species_name'].value_counts())
        # category encoder
        le = preprocessing.LabelEncoder()
        le.fit(df['category_id'].values)
        # family encoder 
        lef = preprocessing.LabelEncoder()
        lef.fit(df['family_name'].values)
        # genus encoder
        leg = preprocessing.LabelEncoder()
        leg.fit(df['genus_name'].values)
        # species encoder
        les = preprocessing.LabelEncoder()
        les.fit(df['species_name'].values)

        # import pdb; pdb.set_trace()
        # np.save('/users/mvaishn1/data/data/mvaishn1/fossil/submission_herb22/onehot.npy', cls_num_list)

        print(nb_classes)
        
        Y_family = lef.transform(df['family_name'].values)
        Y_genus = leg.transform(df['genus_name'].values)
        Y_species = les.transform(df['species_name'].values)
        X_Train, Y_Train = df['image_dir'].values, le.transform(df['category_id'].values)
        # change to species level or genus level hard triplet loss:
        train_full_data = GetDataHierSpecies(train_dir, X_Train, Y_Train, Y_genus, transform, df.index.values, Y_family)

        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), test_size=.12, stratify=Y_Train)
        print(len(val_idx))
        print(len(train_idx))
        train_data = Subset(train_full_data, train_idx) # train_full_data #
        val_data = Subset(train_full_data, val_idx)

        # TEST DATA:
        test_dir = Base_DIR + "test_images/"
        with open(Base_DIR + 'test_metadata.json', "r", encoding="ISO-8859-1") as file:
            test_meta = json.load(file)
        test_ids = [image['image_id'] for image in test_meta]
        test_dirs = [test_dir + image['file_name'] for image in test_meta]
        df_test = pd.DataFrame({
                                "test_id" : test_ids,
                                "test_dir" : test_dirs
                            })
        # print(len(df_test))
        NUM_CL_test = len(df_test['test_dir'].value_counts())
        print(NUM_CL_test)
        X_Test = df_test['test_dir'].values
        Y_Test = None
        test_data =GetData(test_dir, X_Test, Y_Test, transform)

        if is_train=='train':
            dataset = train_data
        elif is_train=='val':
            dataset = val_data
        elif is_train=='test':
            dataset = val_data
        else:
            print('Wrong Dataset')
            exit()
    
    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        # size = 448
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
