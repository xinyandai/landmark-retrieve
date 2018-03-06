import torch.utils.data as data
from PIL import Image
import os
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(file_path):
    return Image.open(file_path).convert('RGB')


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)

    if fv.size == 0:
        return np.zeros((0, 0))

    dim = fv.view(np.int32)[0]
    assert dim > 0

    fv = fv.reshape(-1, 1 + dim)

    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)

    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()

    return fv


class FvecsDataSet(data.Dataset):
    def __init__(self, fvecs_file, transform=None, target_transform=None):
        super(FvecsDataSet, self).__init__()

        self.transform = transform
        self.target_transform = target_transform

        self.features = fvecs_read(fvecs_file)

    def __getitem__(self, index):
        input_feature = self.features[index]
        if self.transform is not None:
            input_feature = self.transform(input_feature)
        if self.target_transform is not None:
            index = self.target_transform(index)
        return input_feature, index

    def __len__(self):
        return len(self.features)


class SingleFolderDataSet(data.Dataset):
    def __init__(self, image_dir, transform=None, target_transform=None, read=True):
        super(SingleFolderDataSet, self).__init__()

        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        self.transform = transform
        self.target_transform = target_transform
        self.read = read

    def __getitem__(self, index):

        if not self.read:
            return self.image_files[index]

        image_file = self.image_files[index]
        input_image = load_img(image_file)

        if self.transform is not None:
            input_image = self.transform(input_image)

        if self.target_transform is not None:
            image_file = self.target_transform(image_file)

        return input_image, image_file

    def __len__(self):
        return len(self.image_files)