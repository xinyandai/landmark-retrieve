from torchvision import models
from torch.autograd import Variable
import torch
import numpy as np

from features.features import features_saver
from features.auto_encoder_train import AutoEncoder


def auto_encoder_features(image_dir, feature_file, model_path, if_gpu=True, input_size=(224, 224)):
    feature_dimension = 512
    auto_encoder = AutoEncoder()
    auto_encoder.load_state_dict(torch.load(model_path))

    alex = models.alexnet(pretrained=True)
    if if_gpu:
        alex.cuda()
        auto_encoder.cuda()

    def extractor(index, data, picture_name):
        """
        :param index: one line data's index
        :param data: one line data
        :param picture_name:
        :return: one line feature of 'data', dimension=1
        """
        nonlocal if_gpu
        nonlocal alex
        nonlocal auto_encoder

        data_var = Variable(data)
        if if_gpu:
            data_var = data_var.cuda()

        cnn_features = alex.features(data_var)
        cnn_features = cnn_features.view(cnn_features.size(0), 256 * 6 * 6)

        encoder_features, decoder_features = auto_encoder.forward(cnn_features)
        encoder_features = encoder_features.data.cpu().numpy()
        return np.resize(encoder_features, [-1])

    features_saver(feature_dimension, extractor, image_dir, feature_file, input_size)