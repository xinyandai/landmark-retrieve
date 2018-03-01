# landmark-retrieve
Given an image, find all of the same landmarks in a dataset.

1. Exact Nearest Neighbor with CNN features (acc:0.014, rank 10/48)
    * use pre-trained AlexNet extract features for train and test data
    * for each test data's feature, find k nearest neighbor in train data
    
