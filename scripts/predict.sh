#!/bin/bash -x

DATASET=${1:-nikon}
MODEL_DIR=saved_models4/${DATASET}
PATH_DATASET_ALL_CSV="/scratch/jagadish/pytorch_fnet/data/csvs/${DATASET}/train.csv"
PATH_RESULTS="/scratch/jagadish/pytorch_fnet/data/results/${DATASET}"
N_IMAGES=20
GPU_IDS=${2:-0}

for TEST_OR_TRAIN in test 
do
    python predict.py \
	   --path_model_dir ${MODEL_DIR} \
	   --path_dataset_csv ${PATH_DATASET_ALL_CSV} \
	   --n_images ${N_IMAGES} \
	   --no_prediction_unpropped \
	   --path_save_dir ${PATH_RESULTS} \
	   --gpu_ids ${GPU_IDS} 
done
