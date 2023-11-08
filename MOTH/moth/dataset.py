import csv
import numpy as np
import os
import torch.utils.data as data

from PIL import Image
from torchvision import datasets

class CelebA_attr(data.Dataset):
    def __init__(self, data_root, train, transforms):
        self.split = 'train' if train else 'test'
        self.dataset = datasets.CelebA(root=data_root, split=self.split,
                                       target_type='attr', download=False)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1)\
                    + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


class GTSRB(data.Dataset):
    def __init__(self, data_root, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(data_root, 'GTSRB/Train')
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(data_root, 'GTSRB/Test')
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + '/' + format(c, '05d') + '/'
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = np.array(images)[indices]
        labels = np.array(labels)[indices]
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, 'GT-final_test.csv')
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + '/' + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label
