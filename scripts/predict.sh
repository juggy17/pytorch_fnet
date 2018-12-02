#!/bin/bash -x

DATASET=${1:-nikon_mito}
MODEL_DIR=saved_models5/${DATASET}
PATH_DATASET_ALL_CSV="/scratch/jagadish/LabelFree/data/csvs/${DATASET}/test.csv"
PATH_RESULTS="/scratch/jagadish/LabelFree/data/results/${DATASET}"
N_IMAGES=10
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
