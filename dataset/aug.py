import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.pytorch.transforms 

from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose,
    RandomRotate90,
    ShiftScaleRotate,
    Blur,
    RandomContrast,
    GaussNoise,
      CLAHE,
      GaussianBlur,
      RandomBrightnessContrast,
      RandomGamma,
      RGBShift,
      HueSaturationValue,
      Normalize
)

# data augumentations
def get_aug(aug, min_area=0., min_visibility=0.25):
    return Compose(aug, bbox_params=BboxParams(format='coco', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['category_id']))

