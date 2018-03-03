from torchvision import models
from torch.autograd import Variable
import numpy as np

from features.features import features_saver


def cnn_features(image_dir, feature_file, if_gpu=True, input_size=(224, 224)):
    feature_dimension = 1000
    alex = models.alexnet(pretrained=True)
    if if_gpu:
        alex.cuda()

    def extractor(index, data, picture_name):
        """
        :param index: one line data's index
        :param data: one line data
        :param picture_name:
        :return: one line feature of 'data', dimension=1
        """
        nonlocal if_gpu
        nonlocal alex

        data_var = Variable(data)
        if if_gpu:
            data_var = data_var.cuda()

        data = alex.forward(data_var)
        # data = model.features(data_var)

        data = data.data.cpu().numpy()
        data = np.resize(data, [-1])
        return data

    features_saver(feature_dimension, extractor, image_dir, feature_file, input_size)