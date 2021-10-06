# Pytorch Lightning Utils
import pytorch_lightning as pl

# Pytorch 
import torch
from torch.utils.data import Dataset, DataLoader

# ML Utils

import cv2
import numpy as np
import pandas as pd
from PIL import Image

#extras
import yaml

# local modules
from src.transforms import *

import torchvision.utils as vutils
import matplotlib.pyplot as plt
    
import yaml
path = "./config.yaml"
Config = yaml.load(open(path), Loader=yaml.FullLoader)


class PawpularDataset(Dataset):
    def __init__(self, image_paths, dense_features, targets, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        features = self.dense_features[item, :]
        
        if self.targets is not None:
            return {
            'image': image.float(),
             #'image': torch.tensor(image, dtype=torch.float),
            'features': torch.tensor(features, dtype=torch.float),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }
        else:
            return {
            'image': torch.tensor(image, dtype=torch.float),
            'features': torch.tensor(features, dtype=torch.float),
        }

        
def train_collate_fn(data):
    
    images = torch.zeros((len(data), 3, Config['img_size'], Config['img_size']))
    datafeatures = torch.zeros((len(data), 12))
    scores = torch.zeros((len(data), 1))
    
    for i in range(len(data)):
        images[i, ...] = data[i]['image']
        datafeatures[i, ...] = data[i]['features']
        scores[i, ...] = data[i]['targets']
        
    return images.float(), datafeatures.float(), scores.float()

def test_collate_fn(data):
    
    images = torch.zeros((len(data), 3, Config['img_size'], Config['img_size']))
    datafeatures = torch.zeros((len(data), 12))

    for i in range(len(data)):
        images[i, ...] = data[i]['image']
        datafeatures[i, ...] = data[i]['features']
        
    return images.float(), datafeatures.float()


# Pytorch lightning Data Module
class PawpularityDModule(pl.LightningDataModule):

    def __init__(self, df, fld, test):
        super().__init__()
        self.fold = fld
        self.train_data = df[df['kfold'] != self.fold].reset_index(drop=True)
        self.val_data = df[df['kfold'] == self.fold].reset_index(drop=True)
        self.test_data = test

        self.dense_features = [
           'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
        ]
        self.train_img_paths = [f"./data/train/{x}.jpg" for x in self.train_data["Id"].values]
        self.valid_img_paths = [f"./data/train/{x}.jpg" for x in self.val_data["Id"].values]
        self.test_img_paths = [f"./data/test/{x}.jpg" for x in self.test_data["Id"].values]

    def setup(self, stage=None):
        self.train_dataset = PawpularDataset(
            image_paths=self.train_img_paths,
            dense_features=self.train_data[self.dense_features].values,
            targets=self.train_data.Pawpularity.values,
            augmentations = get_train_transforms(Config)
            )
        
        self.valid_dataset = PawpularDataset(
            image_paths= self.valid_img_paths,
            dense_features=self.val_data[self.dense_features].values,
            targets = self.val_data.Pawpularity.values,
            augmentations = get_valid_transforms(Config)
        )

        self.test_dataset = PawpularDataset(
            image_paths= self.test_img_paths,
            dense_features=self.test_data[self.dense_features].values,
            targets = None,
            augmentations = get_test_transforms(Config)
        )

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=Config['batch_size'],
                shuffle=True,
                num_workers=Config['num_workers'],
                pin_memory=True, 
                collate_fn = train_collate_fn
            )

    def val_dataloader(self):
        return DataLoader(
                self.valid_dataset,
                batch_size = Config['batch_size'],
                shuffle=False,
                num_workers = Config['num_workers'],
                pin_memory=True,
                collate_fn = train_collate_fn
            )

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset, 
                batch_size = Config['batch_size'],
                shuffle=False, 
                num_workers = Config['num_workers'],
                pin_memory=True,
                collate_fn = test_collate_fn
            )


if __name__ == '__main__':
    
    train = pd.read_csv('./data/train_10folds.csv')
    test = pd.read_csv('./data/train_10folds.csv')
    fold = 0
    dm = PawpularityDModule(train, fold, test)
    dm.setup()

    img, ftrs, targets = next(iter(dm.train_dataloader()))
    
    plt.figure(figsize=(16, 8))
    plt.axis("off")
    plt.title("Training Images", fontsize=30)
    _ = plt.imshow(vutils.make_grid(
        img[:12], nrow=4, padding=7, normalize=True).cpu().numpy().transpose((1, 2, 0)))

    print(f"{g_} {targets[:12].reshape((3,4))} {res}\n\n")
    plt.show()

    img, ftrs = next(iter(dm.test_dataloader()))


    
    