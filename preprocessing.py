import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged, RandCropByPosNegLabeld,
    ToTensord
)

CT_Transforms = Compose([
    LoadImaged(keys=['Image', 'Label']),
    EnsureChannelFirstd(keys=['Image', 'Label']),
    Orientationd(keys=['Image', 'Label'], axcodes='RAS'),
    Spacingd(keys=['Image', 'Label'], pixdim=(1.5, 1.5, 1.5), mode=('bilinear')),
    ScaleIntensityRanged(
        keys=['Image'],
        a_min=-400, a_max=400,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    RandCropByPosNegLabeld(
        keys=['Image', 'Label'],
        label_key='Label',
        spatial_size=(64, 64, 64),
        pos=1,
        neg=0,
        num_samples=5,
        image_key='Image',
        image_threshold=0
    ),
    ToTensord(keys=['Image', 'Label'])
])