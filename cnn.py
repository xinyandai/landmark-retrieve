#! python3
from features.cnn_features import cnn_features
import os

if_gpu = True


if __name__ == '__main__':
    features_name = 'landmark_cnn'
    os.system('mkdir ./data/%s' % features_name)
    # test data features
    path = '/home/xinyan/programs/data/landmark/test'
    filename = './data/%s/%s_query' % (features_name, features_name)
    cnn_features(path, filename, if_gpu=if_gpu)
    # train data features
    path = '/home/xinyan/programs/data/landmark/train'
    filename = './data/%s/%s_base' % (features_name, features_name)
    cnn_features(path, filename, if_gpu=if_gpu)


