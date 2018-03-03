from images.loader import SingleFolderDataSet
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import struct
import os


def features_saver(dimension, feature_extractor, image_dir, feature_file, input_size):
    """
    write features into feature_file + '.fvecs'
    write features into feature_file + '.txt'
    write indexes(integer id and image file name) into feature_file + '_index.txt'
    :param dimension:
    :param feature_extractor: function to extract feature from one line of data
    :param image_dir:
    :param feature_file: dimension/size of feature returned by feature_extractor
    :param input_size: input image's size  of feature_extractor's
    :return: None
    """

    dimension = [dimension]

    data_set = SingleFolderDataSet(image_dir, transform=transforms.Compose([
        transforms.Resize(size=input_size),
        transforms.ToTensor(),
    ]))

    fvecs_file = open(feature_file + '.fvecs', "wb")
    texts_file = open(feature_file + '.txt', 'w')
    index_file = open(feature_file + '_index.txt', 'w')

    train_loader = DataLoader(data_set)

    for batch_idx, (data_feature, (picture_name,)) in enumerate(train_loader):

        data_feature = feature_extractor(batch_idx, data_feature, picture_name)

        fvecs_file.write(struct.pack('i' * len(dimension), *dimension))
        fvecs_file.write(struct.pack('f' * len(data_feature), *data_feature))

        for i in data_feature:
            texts_file.write(str(i) + " ")
        texts_file.write('\n')

        index_file.write("%s,%s\n" % (batch_idx, os.path.basename(picture_name).split(sep='.')[0]))

        if batch_idx % 100 == 0:
            print("index: %10d <----> %1.6f" % (batch_idx, batch_idx / float(len(data_set))))

    fvecs_file.close()
    texts_file.close()
    index_file.close()
