#! python3
from features.cnn_features import cnn_features

if_gpu = True


if __name__ == '__main__':

    path = '/home/xinyan/programs/data/landmark/test'
    filename = './data/test_cnn_features'
    cnn_features(path, filename, if_gpu=if_gpu)

