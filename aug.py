import cv2
import numpy as np
import math
import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.pytorch.transforms 

# define augumentation
def get_train_aug(config):
    train_size = config.data["train_size"]
    if config.data["train_size"]==config.data["input_size"]:
        aug = get_aug([
              A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                         val_shift_limit=0.2, p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.2, 
                                               contrast_limit=0.2, p=0.9),]),
              A.HorizontalFlip(p=0.5),
              A.VerticalFlip(p=0.5),
              A.Resize(train_size, train_size, p=1),
              A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
              ToTensorV2(p=1.0),])
    else:
        # define augumentation
        aug = get_aug([
              A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
              A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                         val_shift_limit=0.2, p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.2, 
                                               contrast_limit=0.2, p=0.9),]),
              A.HorizontalFlip(p=0.5),
              A.VerticalFlip(p=0.5),
              A.RandomCrop(train_size, train_size, p=1),
              A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
              ToTensorV2(p=1.0),])
    return aug

# define augumentation
def get_val_aug(config):
    train_size = config.data["train_size"]
    if config.data["train_size"]==config.data["input_size"]:
        valaug = get_aug([
             A.Resize(train_size, train_size,p=1),
             ToTensorV2(p=1.0),])
    else:
        valaug = get_aug([
             ToTensorV2(p=1.0),])
    return valaug

# data augumentations
def get_aug(aug, min_area=0., min_visibility=0):
    return A.Compose(aug, bbox_params=A.BboxParams(format='coco', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['labels']))