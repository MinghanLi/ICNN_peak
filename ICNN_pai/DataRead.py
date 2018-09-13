from torchvision.datasets import ImageFolder
import csv
import os
import re
import numpy as np
import pandas as pd
from os import getcwd
from os.path import join
from PIL import Image
import torch.utils.data as data
from tqdm import tqdm
from torchvision import transforms

# transform = transforms.Compose([
#                       transforms.Resize(224),
#                       transforms.CenterCrop(224),
#                       transforms.ToTensor()
# ])


class MuraDataset(data.Dataset):

    def __init__(self, csv_f, dataset, transform=None):
        self.dataset = dataset
        if self.dataset == 'Luna16':
            self.df = pd.read_csv(csv_f, names=['img', 'label', 'peak_point', 'diameter'], header=None)
        else:
            self.df = pd.read_csv(csv_f, names=['img', 'label'], header=None)
        self.imgs = self.df.img.values.tolist()
        self.labels = self.df.label.values.tolist()
        # following datasets/folder.py's weird convention here...
        self.samples = [tuple(x) for x in self.df.values]
        self.classes = np.unique(self.labels)
        # number of unique classes
        self.balanced_weights = self.balance_class_weights()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def balance_class_weights(self):
        count = np.zeros(len(self.classes))
        count[0] = len(self.labels)-sum(self.labels)
        count[1] = sum(self.labels)
        weight_per_class = np.divide(sum(count), count)
        weight = np.ones(len(self.labels))
        # for i in range(len(self.labels)):
        #     weight[i] = weight_per_class[self.labels[i]]
        return weight

    def __getitem__(self, idx):
        if self.dataset == 'Luna16':
            img_filename = join('/data/volume1', self.imgs[idx] + '.jpg')
        else:
            img_filename = join('/data/volume1', self.imgs[idx].strip('MURA-v1.1').strip('/').strip(';'))
        image = Image.open(img_filename).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        meta_data = {
            'y_true': label,
            'img_filename': img_filename
        }
        return image, label, meta_data


# data_dir = join(getcwd(), 'MURA-v1.1')
# train_dir = join(getcwd(), 'MURA-v1.1/valid/XR_SHOULDER')
# valid_csv = join('/Users/suqi.lmh/weaksup/assets/datasets/cam_mura/CAM_baseline/MURA_paths', 'valid_shoulder.csv')
# dataset = MuraDataset(valid_csv, transform=transform)
# val_loader = data.DataLoader(dataset,
#                              batch_size=4,
#                              shuffle=False,
#                              num_workers=1,
#                              pin_memory=False)

# pbar2 = tqdm(val_loader)
# for i, (imgs, labels, meta) in enumerate(pbar2):
#     info = meta
#     labels = labels.squeeze()



