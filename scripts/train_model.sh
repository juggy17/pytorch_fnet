#!/bin/bash -x

DATASET=${1:-nikon_mito}
N_ITER=20000
BUFFER_SIZE=30
BATCH_SIZE=24
RUN_DIR="saved_models5/${DATASET}"
PATH_DATASET_ALL_CSV="/scratch/jagadish/LabelFree/data/csvs/${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="/scratch/jagadish/LabelFree/data/csvs/${DATASET}/train.csv"
GPU_IDS=${2:-0}
cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

#python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "/scratch/jagadish/pytorch_fnet/data/csvs" --train_size 0.75 -v
python train_model.py \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency 2000000 \
       --batch_size ${BATCH_SIZE} \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}
