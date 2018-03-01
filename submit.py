#! python3
import numpy as np

nn_result = '../data/landmark_groundtruth.lshbox'


def read_indexes(file_name):
    indexes = []
    # read without new line
    for line_text in open(file_name, 'r').read().splitlines():
        indexes.append(line_text.split(',')[-1])
    return indexes


def extract_ids(line_txt):
    ids = []
    numbers = line_txt.split('\t')
    for i in range(100):
        ids.append(int(numbers[2 * i + 1]))
    return ids


def read_nn(file_name):
    nn = None
    for line_txt in open(file_name):
        if nn is None:
            nn = []
        else:
            nn.append(extract_ids(line_txt))
    return nn


def submission_map(nn, query_indexes, train_indexes):
    nn_map = {}
    train_indexes = np.array(train_indexes)

    for i, top_k in enumerate(nn):
        nn_map[query_indexes[i]] = ' '.join(train_indexes[top_k])

    return nn_map


def replace_submission(sample_sub_file, nn_array, submission_file, query_indexes, train_indexes):
    submission = open(submission_file, 'w')
    sample_sub = open(sample_sub_file)
    sub_map = submission_map(nn_array, query_indexes, train_indexes)
    for line_txt in sample_sub:
        line_array = line_txt.split(',')
        if line_array[0] in sub_map:
            submission.write('%s,%s\n' % (line_array[0], sub_map[line_array[0]]))
        else:
            submission.write(line_txt)

    submission.close()
    sample_sub.close()


if __name__ == '__main__':
    submission_file = '../data/sub.csv'
    sample_submission = '../data/sample_submission.csv'
    query_indexes_file = '../data/test_features_index.txt'
    train_indexes_file = '../data/train_features_index.txt'

    replace_submission(
        sample_submission,
        read_nn(nn_result),
        submission_file,
        read_indexes(query_indexes_file),
        read_indexes(train_indexes_file))