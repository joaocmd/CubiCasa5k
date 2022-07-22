import os
import cv2
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import random

from pathlib import Path

door_classes = (
    'slide', 'rollup', 'none',
    'double_d', 'double_u',
    'opposite_ul_dl', 'opposite_ul_dr', 'opposite_ur_dl', 'opposite_ur_dr',
    'single_dl', 'single_dr', 'single_ul', 'single_ur',
)
rot_classes = tuple(range(0, 360, 45))
symbol_classes = ('corner', )

classes = door_classes + rot_classes + symbol_classes

class_map = {
    'Door Fold Beside': classes.index('slide'),
    'Door None Beside': classes.index('none'),
    'Door ParallelSlide Beside': classes.index('slide'),
    'Door RollUp Beside': classes.index('rollup'), 
    'Door Slide Beside': classes.index('slide'),
    'Door Zfold Beside': classes.index('slide'),
    'Double_d': classes.index('double_d'),
    'Double_u': classes.index('double_u'),
    'Opposite_ul_dl': classes.index('opposite_ul_dl'),
    'Opposite_ul_dr': classes.index('opposite_ul_dr'),
    'Opposite_ur_dl': classes.index('opposite_ur_dl'),
    'Opposite_ur_dr': classes.index('opposite_ur_dr'),
    'Single_dl': classes.index('single_dl'),
    'Single_dr': classes.index('single_dr'),
    'Single_ul': classes.index('single_ul'),
    'Single_ur': classes.index('single_ur'),
}

def getlabel(file_name):
    return file_name.split('-')[1][:-4]
    
def load_image(img_path, img_size):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    diff = height-width
    if diff > 0:
        image = cv2.copyMakeBorder(image, 0, 0, diff//2, diff//2, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    elif diff < 0:
        image = cv2.copyMakeBorder(image, -diff//2, -diff//2, 0, 0, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return image

class RandomResizedCenterCrop(torch.nn.Module):
    def __init__(self, max_crop_ratio=0.8):
        super().__init__()
        self.max_crop_ratio = max_crop_ratio

    def forward(self, img):
        _, H, W = img.size()
        size = torch.LongTensor([H, W])

        ratio = torch.rand(1) * (1-self.max_crop_ratio) + self.max_crop_ratio
        crop_size = (size * ratio).int().tolist()

        img = TF.center_crop(img, crop_size)
        img = TF.resize(img, size.tolist(), interpolation=transforms.InterpolationMode.BILINEAR)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_crop_ratio={self.max_crop_ratio})"

# class Doors(data.Dataset):
#     def __init__(self, split, classes, img_size=40):
#         self.folder = f'../doors/{split}/'
#         self.files = os.listdir(self.folder)
#         self.classes = {c: idx for idx, c in enumerate(classes)}
#         self.img_size = img_size
#         self.split = split

#         self.transform = transforms.Compose([
#             transforms.RandomRotation(7.5),
#             transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
#         ])

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         name = self.files[idx]
#         label = getlabel(name)
#         img = load_image(self.folder + name, self.img_size)
#         img = np.moveaxis(img, 2, 0)
#         img = (img / 255) * 2 - 1
#         if (self.split == 'train'):
#             return self.transform(torch.tensor(img).float()), class_map[label]
#         else:
#             return torch.tensor(img).float(), class_map[label]

# class ToiletsTrain(data.Dataset):
#     def __init__(self, classes, split='train', img_size=100):
#         self.classes = classes
#         self.folder = f'../toilets/{split}/'
#         self.files = os.listdir(self.folder)[1:] # remove A_wrong
#         self.img_size = img_size
#         self.split = split

#         self.transforms = transforms
#         self.transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
#         ])
#         self.resize = RandomResizedCenterCrop(0.75)

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         name = self.files[idx]
#         img = load_image(self.folder + name, self.img_size)
#         img = np.moveaxis(img, 2, 0)
#         img = (img / 255) * 2 - 1
#         img = torch.tensor(img).float()
#         img = self.transform(img)

#         rot = random.choice(self.classes)
#         img = TF.rotate(img, rot, interpolation=transforms.InterpolationMode.BILINEAR, fill=1)

#         img = self.resize(img)

#         return img, rot//(360//len(self.classes))

# class ToiletsEval(data.Dataset):
#     def __init__(self, classes, split, n_samples=None, img_size=100, seed=0):
#         self.classes = classes
#         self.folder = f'../toilets/{split}/'
#         self.files = os.listdir(self.folder)[1:] # remove A_wrong
#         self.files = [f+'@'+str(c) for f in self.files for c in classes]
#         if n_samples != None:
#             random.seed(seed)
#             self.files = random.sample(self.files, n_samples)
#         self.img_size = img_size
#         self.split = split

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         name = self.files[idx]
#         name, rot = name.split('@')
#         rot = int(rot)

#         img = load_image(self.folder + name, self.img_size)
#         img = np.moveaxis(img, 2, 0)
#         img = (img / 255) * 2 - 1
#         img = torch.tensor(img).float()

#         img = TF.rotate(img, rot, interpolation=transforms.InterpolationMode.BILINEAR, fill=1)

#         return img, rot//(360//len(self.classes))

# class SinksTrain(data.Dataset):
#     def __init__(self, classes, split='train', img_size=100, ratio=0.4):
#         self.classes = classes
#         self.folder = f'../sinks/{split}/'
#         self.files = os.listdir(self.folder)[1:] # remove A_wrong
#         self.normal_files = [f for f in self.files if 'Corner' not in f]
#         self.corner_files = [f for f in self.files if 'Corner' in f]
        
#         # let x be the factor we need to multiply the number of elements in c
#         # to meet the ratio between classes
#         # c * x = ratio * (c*x + n)
#         # by solving for x we get:
#         # x = (ratio*n)/((1-ratio)*c)

#         if ratio != None:
#             n = len(self.files)
#             c = len(self.corner_files)
#             needed_factor = (ratio*n)/((1-ratio)*c)
#             self.corner_files = self.corner_files*int(needed_factor)

#         self.sink_files = self.normal_files + self.corner_files
#         self.img_size = img_size

#         self.transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
#         ])
#         self.resize = RandomResizedCenterCrop(0.75)

#     def __len__(self):
#         return len(self.sink_files)

#     def __getitem__(self, idx):
#         name = self.sink_files[idx]
#         img = load_image(self.folder + name, self.img_size)
#         img = np.moveaxis(img, 2, 0)
#         img = (img / 255) * 2 - 1
#         img = torch.tensor(img).float()
#         img = self.transform(img)

#         rot = random.choice(self.classes[:-1])
#         img = TF.rotate(img, rot, interpolation=transforms.InterpolationMode.BILINEAR, fill=1)

#         img = self.resize(img)

#         return img, len(self.classes) - 1 if 'Corner' in name else rot//(360//(len(self.classes)-1))

# class SinksEval(data.Dataset):
#     def __init__(self, classes, split, n_samples=None, img_size=100, seed=0, ratio=0.3):
#         self.classes = classes
#         self.folder = f'../sinks/{split}/'
#         self.files = os.listdir(self.folder)[1:] # remove A_wrong
#         self.normal_files = [f for f in self.files if 'Corner' not in f]
#         self.corner_files = [f for f in self.files if 'Corner' in f]

#         if ratio != None:
#             n = len(self.files)
#             c = len(self.corner_files)
#             needed_factor = (ratio*n)/((1-ratio)*c)
#             self.corner_files = self.corner_files*int(needed_factor)

#         self.sink_files = self.normal_files + self.corner_files       

#         self.sink_files = [f+'@'+str(c) for f in self.files for c in classes[:-1]]


#         if n_samples != None:
#             random.seed(seed)
#             self.sink_files = random.sample(self.sink_files, n_samples)

#         self.img_size = img_size
#         self.split = split

#     def __len__(self):
#         return len(self.sink_files)

#     def __getitem__(self, idx):
#         name = self.sink_files[idx]
#         name, rot = name.split('@')
#         rot = int(rot)

#         img = load_image(self.folder + name, self.img_size)
#         img = np.moveaxis(img, 2, 0)
#         img = (img / 255) * 2 - 1
#         img = torch.tensor(img).float()

#         img = TF.rotate(img, rot, interpolation=transforms.InterpolationMode.BILINEAR, fill=1)

#         return img, len(self.classes) - 1 if 'Corner' in name else rot//(360//(len(self.classes)-1))

# class TrainSet(data.Dataset):
    
#     def __init__(self, split='train', image_size=80):
        
#         doors = Doors(classes[:13], split=split, image_size=image_size)
#         toilets = ToiletsTrain(classes[13:13+8], split=split, image_size=image_size)
#         toilets = SinksTrain(classes[13:13+9], split=split, image_size=image_size)


class TrainSet(data.Dataset):
    def __init__(self, split='train', img_size=80, corner_sinks_ratio=0.4):
        self.img_size = img_size
        self.split = split

        folder = f'doors/{split}'
        self.door_files = [f'{folder}/{f}' for f in os.listdir(folder)]

        self.door_transform = transforms.Compose([
            transforms.RandomRotation(7.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        ])

        folder = f'toilets/{split}'
        self.toilet_files = [f'{folder}/{f}' for f in os.listdir(folder)[1:]] # remove A_wrong

        folder = f'sinks/{split}'
        sink_files = [f'{folder}/{f}' for f in os.listdir(folder)[1:]] # remove A_wrong

        normal_files = [f for f in sink_files if 'Corner' not in f]
        corner_files = [f for f in sink_files if 'Corner' in f]
        
        if corner_sinks_ratio != None:
            n = len(sink_files)
            c = len(corner_files)
            needed_factor = (corner_sinks_ratio*n)/((1-corner_sinks_ratio)*c)
            corner_files = corner_files*int(needed_factor)

        self.sink_files = normal_files + corner_files

        self.symbol_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        ])
        self.resize = RandomResizedCenterCrop(0.75)

        self.all_files = self.door_files + self.toilet_files + self.sink_files

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file = self.all_files[idx]
        img = load_image(file, self.img_size)
        img = np.moveaxis(img, 2, 0)
        img = (img / 255) * 2 - 1
        img = torch.tensor(img).float()


        if 'doors' in file:
            label = getlabel(file)
            return self.door_transform(torch.tensor(img).float()), class_map[label]
        else:
            img = self.symbol_transform(img)

            rot = random.choice(rot_classes[:-1])
            img = TF.rotate(img, rot, interpolation=transforms.InterpolationMode.BILINEAR, fill=1)

            img = self.resize(img)

            label = len(rot_classes) if 'Corner' in file else rot//(360//len(rot_classes))
            return img, label + len(door_classes)


t = TrainSet()
