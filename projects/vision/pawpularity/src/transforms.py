from colorama import Fore, Back, Style
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
c_ = Fore.CYAN
y_ = Fore.YELLOW
res = Style.RESET_ALL

# Image Augmentation Library
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

def mixup_data(x, z, y, lam):
    if lam > 0:
        lam = np.random.beta(
            lam, lam
        )
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda()
 

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_z = lam * z + (1 - lam) * z[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_z, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Augmentations
# There is a well known concept called image augmentations in CNN. What augmentation 
# generally does is, it artificially increases the dataset size by subtly modifying the 
# existing images to create new ones (while training). One added advantage of this is:
# omdel becomes more generalized and focuses to find features and representations
# rather than completely overfitting to the training data.


def get_train_transforms(Config):
    return A.Compose([
                        A.Resize(Config['img_size'], Config['img_size']),
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225], 
                            ),
                                    A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7),
            A.ShiftScaleRotate(
                shift_limit = 0.1, scale_limit=0.1, rotate_limit=45, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2,
                val_shift_limit=0.2, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1), p=0.5
            ),
                        ToTensorV2()
                    ])


def get_valid_transforms(Config):
    return A.Compose([
                        A.Resize(Config['img_size'], Config['img_size']),
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                        ToTensorV2(),
                ])

def get_test_transforms(Config):
    return A.Compose([
                    A.Resize(Config['img_size'], Config['img_size']),
                    A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ToTensorV2(),
                ])