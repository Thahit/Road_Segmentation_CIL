import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from numpy import asarray
import albumentations
from albumentations.pytorch import ToTensorV2
import cv2
import random
import warnings# to ignore the annoaying warning:
#UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. 
# This means writing to this tensor will result in undefined behavior. 
# You may want to copy the array to protect its data or make it writable before converting it to a tensor. 
# This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:199.)
warnings.filterwarnings("ignore")


class ImageList(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.transform = ToTensorV2()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        image = asarray(Image.open("data/" + data).convert('RGB'))

        image = self.transform(image=image)['image']
        image = (image - 128) / 128  # centered around zero
        return image, 0


class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, validation=False,alpha=1.0, tfm=None, has_uncert=False):
        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path1,path2) for path1, path2 in _data_list]
        self.alpha = alpha
        self.tfm = tfm #transformations
        self.has_uncert = has_uncert
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,idx):
        data = self.data_list[idx]
        image = Image.open("data/" + data[0]).convert('RGB')
        groundTruth = Image.open("data/" + data[1]).convert('L')
        if self.has_uncert:
            uncertainty = Image.open(("data/" + data[0]).replace('images', 'uncertainty')).convert('L')

        h,w = image.size

        if  h > 400 or w > 400:# img too big
            # Massachusetts Roads Dataset has 1500*1500
            # just rescaling the images produces a wrong scale
            # the images cover x10 the area our images do
            # -> choose a 150 square and resize
            # #we don't sample from the borders
            h = random.randint(50, h-200)
            w = random.randint(250, w-200)
            image = image.crop((h, w, h+150, w+150))
            if self.has_uncert:
                uncertainty = uncertainty.crop((h, w, h+150, w+150))
            groundTruth = groundTruth.crop((h, w, h+150, w+150))

            # this would be w/o upsampling
            # h = random.randint(50, h-450)
            # w = random.randint(250, w-450)
            #image = image.crop((h, w, h+400, w+400))
            #groundTruth = groundTruth.crop((h, w, h+400, w+400))


            #weight = ... # could weight them lower

        image = asarray(image)
        groundTruth = asarray(groundTruth)

        if self.has_uncert:
            uncertainty = asarray(uncertainty)
            groundTruth = np.dstack([groundTruth, uncertainty])

        augmented = self.tfm(image=image, mask=groundTruth)
        image, groundTruth = augmented["image"], augmented["mask"]
        if self.has_uncert:
            uncertainty = groundTruth[:,:,1]
            groundTruth = groundTruth[:,:,0]
            image = torch.concatenate([image, uncertainty[None, ...]], dim=0)

        groundTruth = torch.where(groundTruth>0, 1, 0)
        
        image = (image.to(torch.float)-128)/128 # centered around zero
        return image, groundTruth  
        

def build_dataloader(path_list, validation=False, batch_size=4, num_workers=1, device='cpu', alpha=1, augmented=True,
                     augment_test=False, has_uncert=False):
    if augmented:
        transformations = albumentations.Compose([
            albumentations.augmentations.transforms.GaussNoise(var_limit=30 ,p=.8),
            albumentations.HorizontalFlip(p=0.4), 
            albumentations.VerticalFlip(p=0.4),
            #albumentations.RandomScale(p=0.4),# no
            albumentations.Rotate(border_mode=cv2.BORDER_WRAP, mask_value=0),#BORDER_CONSTANT or BORDER_WRAP
            albumentations.RandomBrightnessContrast(p=0.4),
            #ElasticTransform(mask_value=0),# DONT LIKE IT
            # options when working with different sizes
            albumentations.SmallestMaxSize(400), #img at least 400*400
            albumentations.RandomCrop(400, 400),# img exactly 400*400
            #albumentations.augmentations.crops.transforms.RandomCrop(height=40, width=40, p=1.),#somehow not working
            #albumentations.Normalize(),
            ToTensorV2(),
        ])
        if augment_test:
            dataset = RoadDataset(path_list,validation=validation,alpha=alpha, tfm=transformations)
            data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=(not validation), drop_last = True)
        else:
            val_transformations = albumentations.Compose([
            albumentations.SmallestMaxSize(400), #img at least 400*400
            albumentations.RandomCrop(400, 400),# img exactly 400*400
            ToTensorV2(),
        ])
            dataset = RoadDataset(path_list,validation=validation,alpha=alpha, 
                                  tfm=val_transformations if ((not augment_test) and validation) else transformations, has_uncert=has_uncert)
            data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=(not validation), drop_last = True, num_workers=num_workers)
        
    else:# DONT
        dataset = RoadDataset(path_list,validation=validation,alpha=alpha)
        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=(not validation), drop_last = True)

    return data_loader
