#!/bin/sh

#export CUDA_VERSION='10.1'
#export CUDNN_VERSION='7.6.0'
export TENSORBOARD_DIR=/scratch/bm4g15/data/nalu-stable-exp/tensorboard
export SAVE_DIR=/scratch/bm4g15/data/nalu-stable-exp/saves

#module load python3
#module load gcc/4.9.2
#module load cuda/$CUDA_VERSION
#module load cudnn/v$CUDNN_VERSION-prod-cuda-$CUDA_VERSION

module load conda/py3-latest  
source deactivate
conda activate nalu-env
cd /home/bm4g15/nalu-stable-exp/

export PYTHONPATH=./ 

python3 -u "$@"
