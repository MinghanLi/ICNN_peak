import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
import ast
import torch
import torch.utils.data as data
from torchvision import transforms

transform = transforms.Compose([
                      transforms.Resize(224),
                      transforms.CenterCrop(224),
                      transforms.ToTensor()
])


class MuraDataset(data.Dataset):

    def __init__(self, csv_f, transform=transform):
        self.df = pd.read_csv(csv_f, names=['img', 'labels', 'peak_point', 'diameter'], header=None)
        self.imgs = self.df.img.values.tolist()
        self.labels = self.df.labels.values.tolist()
        self.peak_point = self.df.peak_point.values.tolist()
        # following datasets/folder.py's weird convention here...
        self.samples = [tuple(x) for x in self.df.values]
        self.classes = np.unique(self.labels)
        # number of unique classess
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_filename = join('/data/volume1', self.imgs[idx]+'.jpg')
        image = Image.open(img_filename).convert('RGB')
        label = self.labels[idx]
        peak_point = self.peak_point[idx]

        meta_data = {
            'y_true': label,
            'img_filename': self.imgs[idx],
            'peak_point': torch.Tensor(ast.literal_eval(peak_point)),
            'img_size': torch.Tensor(image.size)
        }

        if self.transform:
            image = self.transform(image)
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


