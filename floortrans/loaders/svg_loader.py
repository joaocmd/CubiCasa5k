import lmdb
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from numpy import genfromtxt
from floortrans.loaders.house import House


class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file, is_transform=True,
                 augmentations=None, img_norm=True, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/',
                 return_heatmaps=False):
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.get_data = None
        self.original_size = original_size
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'
        self.return_heatmaps = return_heatmaps

        if format == 'txt':
            self.get_data = self.get_txt
        if format == 'lmdb':
            self.lmdb = lmdb.open(data_folder+lmdb_folder, readonly=True,
                                  max_readers=8, lock=False,
                                  readahead=True, meminit=False)
            self.get_data = self.get_lmdb
            self.is_transform = True

        self.data_folder = data_folder
        # Load txt file to list
        self.folders = genfromtxt(data_folder + data_file, dtype='str')

    def __len__(self):
        """__len__"""
        return len(self.folders)

    def __getitem__(self, index):
        # sample["image"] size is num_channels x img_width x img_height
        # sample["label"] size is 2 x img_width x img_height (pixel-wise clf)
        # sample["label"][0] is the classification of rooms (12 labels)
        # sample["label"][1] is the classification of the icons (11 labels)
        # Note: 
        # - room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
        # - icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
        sample = self.get_data(index)
        folder = sample['folder']

        if self.get_data == self.get_lmdb:
            sample['image'] = sample['image'].float()
            sample['label'] = sample['label'].float()

        if self.augmentations is not None:
            # Note: the augmentation DictToTensor transforms sample["label"]
            # from 2 x img_width x img_height to 23 x img_width x img_height
            sample = self.augmentations(sample)
            if torch.any(sample['label'][-1] > 10):
                print("Danger! Icons' label above 10 for filepath", folder)
         
            if torch.any(sample['label'][-2] > 11):
                print("Danger! Rooms' label above 11 for filepath", folder)

        if self.is_transform:
            sample = self.transform(sample)

        val = {**sample, 'folder': folder}
        if not self.return_heatmaps:
            del val['heatmaps']
        else:
            # Note: If we end up using the heatmaps as classes, the attribute
            # sample["heatmaps"] will be a dictionary where the keys are one of
            # the following classes, and the values are the pixels location:
            #   0. I
            #   1. I top to right
            #   2. I vertical flip
            #   3. I top to left
            #   4. L horizontal flip
            #   5. L
            #   6. L vertical flip
            #   7. L horizontal and vertical flip
            #   8. T
            #   9. T top to right
            #   10. T top to down
            #   11. T top to left
            #   12. X or +
            #   13. Opening left corner
            #   14. Opening right corner
            #   15. Opening up corner
            #   16. Opening down corer
            #   17. Icon upper left
            #   18. Icon upper right
            #   19. Icon lower left
            #   20. Icon lower right
            # To use it for classification we need to add an extra value for
            # the "non-junction".
            pass
        return val

    def get_txt(self, index):
        fplan = cv2.imread(self.data_folder + self.folders[index] + self.image_file_name)
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
        height, width, nchannel = fplan.shape
        fplan = np.moveaxis(fplan, -1, 0)

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)
        # Combining them to one numpy tensor
        label = torch.tensor(house.get_segmentation_tensor().astype(np.uint8))
        heatmaps = house.get_heatmap_dict()
        coef_width = 1
        if self.original_size:
            fplan = cv2.imread(self.data_folder + self.folders[index] + self.org_image_file_name)
            fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
            height_org, width_org, nchannel = fplan.shape
            fplan = np.moveaxis(fplan, -1, 0)
            label = label.unsqueeze(0)
            label = torch.nn.functional.interpolate(label,
                                                    size=(height_org, width_org),
                                                    mode='nearest')
            label = label.squeeze(0)

            coef_height = float(height_org) / float(height)
            coef_width = float(width_org) / float(width)
            for key, value in heatmaps.items():
                heatmaps[key] = [(int(round(x*coef_width)), int(round(y*coef_height))) for x, y in value]

        img = torch.tensor(fplan.astype(np.uint8))

        sample = {'image': img, 'label': label, 'folder': self.folders[index],
                  'heatmaps': heatmaps, 'scale': coef_width}
        sample['heatmaps'] = {i: [[int(x), int(y)] for x, y in v] for i, v in sample['heatmaps'].items()}

        return sample

    def get_lmdb(self, index):
        key = self.folders[index].encode()
        with self.lmdb.begin(write=False) as f:
            data = f.get(key)

        sample = pickle.loads(data)
        sample['folder'] = key
        if "9243" in self.folders[index]:
            print(key)
        sample['heatmaps'] = {i: [[int(x), int(y)] for x, y in v] for i, v in sample['heatmaps'].items()}

        return sample

    def transform(self, sample):
        fplan = sample['image']
        # Normalization values to range -1 and 1
        fplan = 2 * (fplan / 255.0) - 1

        sample['image'] = fplan

        return sample
