#!/bin/bash
pwd
whoami
nvidia-smi
ifconfig
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-9.0
echo $LD_LIBRARY_PATH
echo $CUDA_HOME
echo "SLURM_JOB_ID" $SLURM_JOB_ID
ulimit -u 131072
# source ~/gpuenv/activate
# cd ~/gitmisc/Mask_RCNN
pwd
ls
git log -1 --pretty=\%B

source /home/jagadish/anaconda3/bin/activate juggy3
cd /home/jagadish/projects/external/pytorch_fnet_gpu/pytorch_fnet/

# copy project files to scratch folder
# rsync -a ./pytorch_fnet/* $LOCAL_SSD
# cd $LOCAL_SSD

echo sh scripts/train_model.sh dna 0

stdbuf -oL sh scripts/train_model.sh dna 0

# copy results
# rsync -a saved_models/TEST/ /home/jagadish/projects/external/pytorch_fnet_gpu/pytorch_fnet/saved_models/TEST/
