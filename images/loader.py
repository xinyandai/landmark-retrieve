import torch.utils.data as data
from PIL import Image
import os


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(file_path):
    return Image.open(file_path).convert('RGB')


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