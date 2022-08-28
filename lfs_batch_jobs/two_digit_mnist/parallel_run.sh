#!/bin/bash

# cmmd: bash parallel_run.sh <start seed> <end seed> <GPU-ID>
# loops over the different folds

logs_base_dir='/data/nalms/logs'
data_path_flag='/datasets/bm4g15/datasets/two-digit-mnist'
no_cuda_flag=''
no_save_flag=''
verbose_flag=''


for seed in $(eval echo {$1..$2})
  do

  export TENSORBOARD_DIR=/data/nalms/tensorboard
  export SAVE_DIR=/data/nalms/saves
  export PYTHONPATH=./


######################################################################################################################
# Division
######################################################################################################################

  experiment_name='two-digit-mnist/div/ID79-1digit_conv-div'
  mkdir -p ${logs_base_dir}/${experiment_name}/errors
  id=79
  CUDA_VISIBLE_DEVICES=$3 python3 -u /home/bm4g15/nalu-stable-exp/experiments/two_digit_mnist.py \
    --seed ${seed} \
   --id ${id} --operation div --max-epochs 1000 \
    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5  \
    > ${logs_base_dir}/${experiment_name}/${seed}.out \
    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

  experiment_name='two-digit-mnist/div/ID78-1digit_conv-nru_R30-40-2_gnc1_s3Init'
  mkdir -p ${logs_base_dir}/${experiment_name}/errors
  id=78
  CUDA_VISIBLE_DEVICES=$3 python3 -u /home/bm4g15/nalu-stable-exp/experiments/two_digit_mnist.py \
    --seed ${seed} \
   --id ${id} --operation div --nalm-name nru --use-nalm --learn-labels2out --max-epochs 1000 \
    --regualizer-scaling-start 30 --regualizer-scaling-end 40 --regualizer 2 --clip-grad-norm 1 \
    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5  \
    > ${logs_base_dir}/${experiment_name}/${seed}.out \
    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

  experiment_name='two-digit-mnist/div/ID80-1digit_conv-realnpu-mod_R30-40-2_s3Init'
  mkdir -p ${logs_base_dir}/${experiment_name}/errors
  id=80
  CUDA_VISIBLE_DEVICES=$3 python3 -u /home/bm4g15/nalu-stable-exp/experiments/two_digit_mnist.py \
    --seed ${seed} \
    --id ${id} --operation div --nalm-name realnpu-mod --use-nalm --learn-labels2out --max-epochs 1000 \
    --regualizer-scaling-start 30 --regualizer-scaling-end 40 --regualizer-npu-w 2 --regualizer-gate 2 \
    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5  \
    > ${logs_base_dir}/${experiment_name}/${seed}.out \
    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

  experiment_name='two-digit-mnist/div/ID81-1digit_conv-nmruSign_R30-40-2_gncF_s3Init'
  mkdir -p ${logs_base_dir}/${experiment_name}/errors
  id=81
  CUDA_VISIBLE_DEVICES=$3 python3 -u /home/bm4g15/nalu-stable-exp/experiments/two_digit_mnist.py \
    --seed ${seed} \
   --id ${id} --operation div --nalm-name nmru-sign --use-nalm --learn-labels2out --max-epochs 1000 \
    --regualizer-scaling-start 30 --regualizer-scaling-end 40 --regualizer 2 \
    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5  \
    > ${logs_base_dir}/${experiment_name}/${seed}.out \
    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err

  experiment_name='two-digit-mnist/div/ID84-1digit_conv-mlp_2L-256HU_wDecay-1e-3'
  mkdir -p ${logs_base_dir}/${experiment_name}/errors
  id=84
  CUDA_VISIBLE_DEVICES=$3 python3 -u /home/bm4g15/nalu-stable-exp/experiments/two_digit_mnist.py \
    --seed ${seed} \
    --id ${id} --operation div --learn-labels2out --max-epochs 1000 \
    --weight-decay 0.001 \
    --data-path ${data_path_flag} ${verbose_flag} ${no_cuda_flag} ${no_save_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --dataset-workers 5  \
    > ${logs_base_dir}/${experiment_name}/${seed}.out \
    2> ${logs_base_dir}/${experiment_name}/errors/${seed}.err


done
wait 

date
echo "Script finished."
