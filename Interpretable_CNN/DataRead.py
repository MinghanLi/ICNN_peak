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
from torchvision import transforms

transform = transforms.Compose([
                      transforms.Resize(224),
                      transforms.CenterCrop(224),
                      transforms.ToTensor()
])


class MuraDataset(data.Dataset):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, csv_f, transform=transform, download=False):
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
        img_filename = join('/data/volume1', self.imgs[idx].strip('MURA-v1.1').strip('/').strip(';')) # strip: remove top/down str';'
        # print(os.listdir('/data/volume1'))
        image = Image.open(img_filename).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        patient = int(self._patient_re.search(img_filename).group(1))
        study = int(self._study_re.search(img_filename).group(1))
        image_num = int(self._image_re.search(img_filename).group(1))
        study_type = self._study_type_re.search(img_filename).group(1)

        meta_data = {
            'y_true': label,
            'img_filename': img_filename,
            'patient': patient,
            'study': study,
            'study_type': study_type,
            'image_num': image_num,
            'encounter': "{}_{}_{}".format(study_type, patient, study)
        }
        return image, label, meta_data


# data_dir = join(getcwd(), 'MURA-v1.1')
# train_dir = join(getcwd(), 'MURA-v1.1/train/XR_SHOULDER')
# train_csv = join(data_dir, 'train_shoulder.csv')
# dataset = MuraDataset(train_csv, transform=transform)
# val_loader = data.DataLoader(dataset,
#                              batch_size=1,
#                              shuffle=False,
#                              num_workers=1,
#                              pin_memory=False)


