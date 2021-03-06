#! python3
from features.auto_ecoder_features import auto_encoder_features
import os
import sys

if_gpu = True


if __name__ == '__main__':
    features_name = 'landmark_encoder'
    test_images = '/home/xinyan/programs/data/landmark/test'
    train_images = '/home/xinyan/programs/data/landmark/train'
    model_path = './features/auto_encoder.torch'

    if len(sys.argv) > 3:
        features_name = sys.argv[0]
        test_images = sys.argv[1]
        train_images = sys.argv[2]
    else:
        print('\033[93m', 'use default features_name test_images_folder train_images_folder')
        print('\033[93m', 'usage: python cnn.py <features_name> <test_images_folder> <train_images_folder>')

    # make directory
    os.system('mkdir ./data/%s' % features_name)

    # test data features extract
    # filename = './data/%s/%s_query' % (features_name, features_name)
    # auto_encoder_features(test_images, filename, model_path, if_gpu=if_gpu)

    # train data features extract
    filename = './data/%s/%s_base' % (features_name, features_name)
    auto_encoder_features(train_images, filename, model_path, if_gpu=if_gpu)


