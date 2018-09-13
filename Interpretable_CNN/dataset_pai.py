
import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import tarfile
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import scipy.io as sio
from subprocess import call
import h5py

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class CUBDataset(MNIST):
    train_transform = test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    def download(self):
        folder = os.path.join(self.root, self.raw_folder)
        folder += '/CUB_200_2011'

        # process and save as torch files
        print('Processing...')

        training_set = []
        testing_set = []

        with open(folder+'/train_test_split.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_id, cat = line.split(' ')
                cat = cat[0]
                if cat == '1':
                    training_set.append(image_id)
                else:
                    testing_set.append(image_id)

        class_dict = {}
        with open(folder+'/classes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id, class_name = line.split(' ')
                class_dict[class_id] = class_name[:-1]

        images_dict = {}
        with open(folder+'/images.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, img_name = line.split(' ')
                images_dict[img_id] = img_name[:-1]

        image_class_dict = {}
        with open(folder+'/image_class_labels.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, class_id = line.split(' ')
                image_class_dict[img_id] = class_id[:-1]

        bbox_dict = {}
        with open(folder+'/bounding_boxes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, x, y, w, h = line.split(' ')
                bbox_dict[img_id] = (float(x), float(y), float(w), float(h[:-1]))

        train_data, train_label = [], []
        test_data, test_label = [], []

        for i, img_id in enumerate(training_set):
            # print(i, img_id)
            x = np.array(Image.open(folder + '/images/' + images_dict[img_id]))
            bbox = tuple(int(x) for x in bbox_dict[img_id])
            if x.shape[-1] == 1:
                x = x.repeat(3, -1)
            x = x[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            train_data.append(torch.from_numpy(x))
            train_label.append(torch.LongTensor([int(image_class_dict[img_id])]))

        for i, img_id in enumerate(testing_set):
            # print(i, img_id)
            x = np.array(Image.open(folder + '/images/' + images_dict[img_id]))
            bbox = tuple(int(x) for x in bbox_dict[img_id])
            x = x[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            test_data.append(torch.from_numpy(x))
            test_label.append(torch.LongTensor([int(image_class_dict[img_id])]))

        training_set = (train_data, train_label)
        test_set = (test_data, test_label)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target - 1
        if img.size(0) == 1:
            img = img.expand(3, 224, 224)

        return img, target


if __name__ == '__main__':
    dataset = CUBDataset('data/CUB', download=True)
    for example in dataset:
        pass
