#! python3

from images.loader import SingleFolderDataSet
from torch.utils.data.dataloader import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision import datasets

from torch.autograd import Variable
import struct

import os

import numpy as np


if_gpu = True


if __name__ == '__main__':

    path = '/home/xinyan/programs/data/landmark/test'
    filename = './data/test_cnn_features'

    data_set = SingleFolderDataSet(path, transform=transforms.Compose([
        transforms.Scale(256),
        transforms.Resize(size=[224, 224]),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

    fvecs_file = open(filename+'.fvecs', "wb")
    txt_file = open(filename+'.txt', 'w')
    index_file = open(filename+'_index.txt', 'w')

    # dimension = [256 * 6 * 6]
    dimension = [1000]

    alex = models.alexnet(pretrained=True)
    if if_gpu:
        alex.cuda()

    train_loader = DataLoader(data_set)

    for batch_idx, (data, (picture_name, )) in enumerate(train_loader):

        data_var = Variable(data)
        if if_gpu:
            data_var = data_var.cuda()

        data = alex.forward(data_var)
        # data = alex.features(data_var)

        data = data.data.cpu().numpy()
        data = np.resize(data, [-1])

        fvecs_file.write(struct.pack('i' * len(dimension), *dimension))
        fvecs_file.write(struct.pack('f' * len(data), *data))

        for i in data:
            txt_file.write(str(i) + " ")
        txt_file.write('\n')

        index_file.write("%s,%s\n" % (batch_idx, os.path.basename(picture_name).split(sep='.')[0]))

        if batch_idx % 100 == 0:
            print("index: %10d <----> %1.6f" % (batch_idx, batch_idx / float(len(data_set))))

    fvecs_file.close()
    txt_file.close()
