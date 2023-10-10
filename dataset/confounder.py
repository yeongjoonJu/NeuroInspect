import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets.utils import download_and_extract_archive
from sklearn.model_selection import train_test_split
from wilds.datasets.fmow_dataset import FMoWDataset

def get_transform_cub(train, augment_data):
    scale = 256.0/224.0
    target_resolution = (224,224)
    # assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_transform_celebA(train, augment_data):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    """
    if model_attributes[model_type]['target_resolution'] is not None:
        target_resolution = model_attributes[model_type]['target_resolution']
    else:
        target_resolution = (orig_w, orig_h)
    """
    target_resolution = (224,224)

    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


class ConfounderDataset(Dataset):
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        img_filename = os.path.join(
            self.data_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
            self.eval_transform):
            img = self.eval_transform(img)

        x = img

        return x,y,g

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac<1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name
    
    def get_img_files(self, splits=["train", "val", "test"]):
        subsets = {}
        for split in splits:
            subsets[split] = []
        
        id2split = {0: "train", 1: "val", 2: "test"}
            
        for idx, filename in enumerate(self.filename_array):
            filename = os.path.join(self.data_dir, filename)
            subsets[id2split[self.split_array[idx]]].append(filename)
                
        return subsets
    

class DRODataset(Dataset):
    def __init__(self, dataset, image_files, process_item_fn, n_classes, n_groups=None, group_str_fn=None):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        self._image_files = image_files
        group_array = []
        y_array = []

        print("Assigning group labels...")
        for _,y,g in tqdm(self):
            group_array.append(g)
            y_array.append(y)
        print("Done.")
        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x,y,g in self:
            return x.size()

    def get_loader(self, train, reweight_groups, **kwargs):
        if not train: # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups: # Training but not reweighting
            shuffle = True
            sampler = None
        else: # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self)/self._group_counts
            weights = group_weights[self._group_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader
    
    
class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """
    
    _URL = "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz"

    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False, download=True):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.data_dir = os.path.join(self.root_dir, "cub/waterbird_complete95_forest2water2/waterbird_complete95_forest2water2")
        
        if download:
            self._download()
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.classes = ["landbird", "waterbird"]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Set transform
        self.features_mat = None
        self.train_transform = get_transform_cub(train=True, augment_data=augment_data)
        self.eval_transform = get_transform_cub(train=False, augment_data=augment_data)
        
    def _check_exists(self) -> bool:
        return os.path.exists(self.data_dir)
        

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root)
        
    
class MetaDatasetCatDog(ConfounderDataset):
    """
    MetaShift data. 
    `cat` is correlated with (`sofa`, `bed`), and `dog` is correlated with (`bench`, `bike`);
    In testing set, the backgrounds of both classes are `shelf`.

    Args: 
        args : Arguments, see run_expt.py
        root_dir (str): Arguments, see run_expt.py
        target_name (str): Data label
        confounder_names (list): A list of confounders
        model_type (str, optional): Type of model on the dataset, see models.py
        augment_data (bool, optional): Whether to use data augmentation, e.g., RandomCrop
        mix_up (bool, optional): Whether to use mixup 
        mix_alpha, mix_unit, mix_type, mix_freq, mix_extent: Variables in LISA implemenation
        group_id (int, optional): Select a subset of dataset with the group id
        
    """
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 mix_up=False,
                 mix_alpha=2,
                 mix_unit='group',
                 mix_type=1,
                 mix_freq='batch',
                 mix_extent=None,
                 group_id=None):
        self.mix_up = mix_up
        self.mix_alpha = mix_alpha
        self.mix_unit = mix_unit
        self.mix_type = mix_type
        self.root_dir = root_dir+"/metashifts/MetaDatasetCatDog"
        self.data_dir = ""
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.RGB = True
        self.n_confounders = 1

        self.train_data_dir = os.path.join(self.root_dir, "train")
        self.test_data_dir = os.path.join(self.root_dir, 'test')

        # Set training and testing environments
        self.n_classes = 2
        self.n_groups = 4
        cat_dict = {0: ["sofa"], 1: ["bed"]}
        dog_dict = {0: ['bench'], 1: ['bike']}
        self.test_groups = { "cat": ["shelf"], "dog": ["shelf"]}
        self.train_groups = {"cat": cat_dict, "dog": dog_dict}
        self.train_filename_array, self.train_group_array, self.train_y_array = self.get_data(self.train_groups,
                                                                                              is_training=True)
        self.test_filename_array, self.test_group_array, self.test_y_array = self.get_data(self.test_groups,
                                                                                           is_training=False)
        # split test and validation set
        np.random.seed(100)
        test_idxes = np.arange(len(self.test_group_array))
        val_idxes, _ = train_test_split(np.arange(len(test_idxes)), test_size=0.85, random_state=0)
        test_idxes = np.setdiff1d(test_idxes, val_idxes)
        
        # define the split array
        self.train_split_array = np.zeros(len(self.train_group_array))
        self.test_split_array = 2 * np.ones(len(self.test_group_array))
        self.test_split_array[val_idxes] = 1

        self.filename_array = np.concatenate([self.train_filename_array, self.test_filename_array])
        self.group_array = np.concatenate([self.train_group_array, self.test_group_array])
        self.split_array = np.concatenate([self.train_split_array, self.test_split_array])
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.y_array = np.concatenate([self.train_y_array, self.test_y_array])
        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(1, torch.tensor(self.y_array).unsqueeze(1), 1).numpy()
        self.mix_array = [False] * len(self.y_array)

        if group_id is not None:
            idxes = np.where(self.group_array == group_id)
            self.filename_array = self.filename_array[idxes]
            self.group_array = self.group_array[idxes]
            self.split_array = self.split_array[idxes]
            self.y_array = self.y_array[idxes]
            self.y_array_onehot = self.y_array_onehot[idxes]

        self.precomputed = False
        self.train_transform = get_transform_cub(train=True, augment_data=augment_data)
        self.eval_transform = get_transform_cub(train=False, augment_data=augment_data)

        self.domains = self.group_array
        self.n_groups = len(np.unique(self.group_array))

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

    def group_str(self, group_idx, train=False):
        if not train:
            if group_idx < len(self.test_groups['cat']):
                group_name = f'y = cat'
                group_name += f", attr = {self.test_groups['cat'][group_idx]}"
            else:
                group_name = f"y = dog"
                group_name += f", attr = {self.test_groups['dog'][group_idx - len(self.test_groups['cat'])]}"
        else:
            if group_idx < len(self.train_groups['cat']):
                group_name = f'y = cat'
                group_name += f", attr = {self.train_groups['cat'][group_idx][0]}"
            else:
                group_name = f"y = dog"
                group_name += f", attr = {self.train_groups['dog'][group_idx - len(self.train_groups['cat'])][0]}"
        return group_name

    def get_data(self, groups, is_training):
        filenames = []
        group_ids = []
        ys = []
        id_count = 0
        animal_count = 0
        for animal in groups.keys():
            if is_training:
                for _, group_animal_data in groups[animal].items():
                    for group in group_animal_data:
                        for file in os.listdir(f"{self.train_data_dir}/{animal}/{animal}({group})"):
                            filenames.append(os.path.join(f"{self.train_data_dir}/{animal}/{animal}({group})", file))
                            group_ids.append(id_count)
                            ys.append(animal_count)
                    id_count += 1
            else:
                for group in groups[animal]:
                    for file in os.listdir(f"{self.test_data_dir}/{animal}/{animal}({group})"):
                        filenames.append(os.path.join(f"{self.test_data_dir}/{animal}/{animal}({group})", file))
                        group_ids.append(id_count)
                        ys.append(animal_count)
                    id_count += 1
            animal_count += 1
        return filenames, np.array(group_ids), np.array(ys)
    

class CelebADataset(ConfounderDataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(self, root_dir, target_name, confounder_names, augment_data):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data

        # Read in attributes
        self.attrs_df = pd.read_csv(
            os.path.join(root_dir, 'celeba', 'list_attr_celeba.csv'))

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root_dir, 'celeba', 'img_align_celeba')
        self.filename_array = self.attrs_df['File_name'].values
        self.attrs_df = self.attrs_df.drop(labels='File_name', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Read in train/val/test splits
        self.split_df = pd.read_csv(
            os.path.join(root_dir, 'celeba', 'list_eval_partition.csv'))
        self.split_array = self.split_df['partition'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        
        self.features_mat = None
        self.train_transform = get_transform_celebA(train=True, augment_data=augment_data)
        self.eval_transform = get_transform_celebA(train=False, augment_data=augment_data)
        
    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)