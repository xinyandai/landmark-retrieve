#!/usr/bin/env bash

GCC_FLAGS="-g"
GCC_LINK="-lpthread"

g++ -I ../../include --std=c++11 ${GCC_FLAGS} src/exact_nn.cpp -o bin/nns ${GCC_LINK}

topk=10
numThreads=20

metric="euclidean"
# metric="angular"
# metric="product"

iter=0

for dataset in "landmark_cnn"
do
    iter=`expr $iter + 1`

    # input data files
    base_file="../data/${dataset}/${dataset}_base.fvecs"
    query_file="../data/${dataset}/${dataset}_query.fvecs"

    # output groundtruth files, support both ivecs and lshbox formats
    ivecs_bench_file="../data/${dataset}/${dataset}_${metric}_groundtruth.ivecs"
    lshbox_bench_file="../data/${dataset}/${dataset}_${metric}_groundtruth.lshbox"

    bin/nns $base_file $query_file $topk $lshbox_bench_file $ivecs_bench_file $metric $numThreads
done
