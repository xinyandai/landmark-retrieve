# landmark-retrieve
Given an image, find all of the same landmarks in a dataset. [kaggle: landmark-retrieval-challenge](https://www.kaggle.com/c/landmark-retrieval-challenge)


## run
* download data set index files
    - data/sample_submission.csv
    - data/index.csv
    - data/test.csv
* download data set from website
    ```
    ## python3 download.py <data_file.csv> <output_dir/>
    
    ## train data
    python3 download.py data/index.csv output_dir/train
    ## test data
    python3 download.py data/test.csv output_dir/test
    
    ```

## retrieve methods

* Exact Nearest Neighbor with CNN features (acc:0.014, rank 10/48)
    * use pre-trained AlexNet extract features for train and test data
    ```
    ## python cnn.py <features_name> <test_images_folder> <train_images_folder>
    python3 cnn.py landmark_cnn output_dir/test output_dir/train
    ```
    * for each test data's feature, find k nearest neighbor in train data
    ```
    cd script
    ## generate k neareset neighbor to ./data/landmark_cnn/landmark_cnn_euclidean_groundtruth.lshbox
    sh nns.sh 
    ```
    * generate submissions
    ```
    ## read k neareset neighbor and replace retrieve result in sample_submission
    ## python3 submit.py <features_name> <submission_file>
    python3 submit.py landmark_cnn sub.csv
    ```
    * submit sub.csv
    
