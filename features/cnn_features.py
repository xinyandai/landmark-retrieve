from torchvision import models
from torch.autograd import Variable
import numpy as np

from features.features import features_saver


def cnn_features(image_dir, feature_file, if_gpu=True, input_size=(224, 224)):
    dimension = 1000
    alex = models.alexnet(pretrained=True)
    if if_gpu:
        alex.cuda()

    def extractor(batch_idx, data, picture_name):
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

    features_saver(dimension, extractor, image_dir, feature_file, input_size)