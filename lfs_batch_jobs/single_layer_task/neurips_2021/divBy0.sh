#!/bin/bash

# cmmd: bash divBy0.sh 0 24
# TODO - make sure log and err log folders exist

#export LSB_DJOB_NUMPROC=14

verbose_flag=''
no_save_flag='--no-save'

ranges=( '[0,1e-4]' '[0,1e-3]' '[0,1e-2]' '[0,1e-1]' '[0,1]' )

for ((i=0;i<${#ranges[@]};++i))
  do
  for seed in $(eval echo {$1..$2})
    do

    export TENSORBOARD_DIR=/data/nalms/tensorboard
    export SAVE_DIR=/data/nalms/saves
    export PYTHONPATH=./

############################################## Easy #########################################################
    # op: reciprocal ->  1 / a where input is [a]

    # Real NPU
    experiment_name='in2_divBy0/easy/realnpu'
    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
    --operation reciprocal --layer-type RealNPU --nac-mul real-npu \
    --input-size 1 --subset-ratio 1 --num-subsets 1 \
    --regualizer-shape none --regualizer 0  --learning-rate 5e-3 \
    --npu-clip wg --npu-Wr-init xavier-uniform-constrained \
    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
    --seed ${seed} --max-iterations 5000 ${verbose_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
    > /data/nalms/logs/in2_divBy0/easy/realnpu/${ranges[i]}-${seed}.out \
    2> /data/nalms/logs/in2_divBy0/easy/realnpu/errors/${ranges[i]}-${seed}.err & # parallel version

    # NRU
#    experiment_name='in2_divBy0/easy/nru'
#    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
#    --operation reciprocal --layer-type NRU --nac-mul mnac \
#    --input-size 1 --subset-ratio 1 --num-subsets 1 \
#    --regualizer-scaling-start 15000 --regualizer-scaling-end 20000 \
#    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
#    --seed ${seed} --max-iterations 5000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1 \
#    > /data/nalms/logs/in2_divBy0/easy/nru/${ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/in2_divBy0/easy/nru/errors/${ranges[i]}-${seed}.err & # parallel version

    # NMRU
#    experiment_name='in2_divBy0/easy/signNMRU'
#    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
#    --operation reciprocal --layer-type SignNMRU \
#    --input-size 1 --subset-ratio 1 --num-subsets 1 \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
#    --seed ${seed} --max-iterations 5000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda --no-save \
#    --clip-grad-norm 1 --learning-rate 1e-2 \
#    > /data/nalms/logs/in2_divBy0/easy/signNMRU/${ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/in2_divBy0/easy/signNMRU/errors/${ranges[i]}-${seed}.err & # parallel version

############################################## Medium #######################################################
    # op: reciprocal -> 1 / a where input is [a,b] and a,b are both drawn from the same uniform range

    # Real NPU
#    experiment_name='in2_divBy0/medium/realnpu'
#    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
#    --operation reciprocal --layer-type RealNPU --nac-mul real-npu \
#    --regualizer-shape none --regualizer 0 --learning-rate 5e-3 \
#    --regualizer-l1 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#    --npu-clip wg --npu-Wr-init xavier-uniform-constrained \
#    --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#    --regualizer-scaling-start 40000 --regualizer-scaling-end 50000 \
#    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/in2_divBy0/medium/realnpu/${ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/in2_divBy0/medium/realnpu/errors/${ranges[i]}-${seed}.err & # parallel version


    # NRU
#    experiment_name='in2_divBy0/medium/nru'
#    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
#    --operation reciprocal --layer-type NRU --nac-mul mnac \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1 \
#    > /data/nalms/logs/in2_divBy0/medium/nru/${ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/in2_divBy0/medium/nru/errors/${ranges[i]}-${seed}.err & # parallel version


    # NMRU
#    experiment_name='in2_divBy0/medium/signNMRU'
#    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
#    --operation reciprocal --layer-type SignNMRU \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda --no-save \
#    --clip-grad-norm 1 --learning-rate 1e-2 \
#    > /data/nalms/logs/in2_divBy0/medium/signNMRU/${ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/in2_divBy0/medium/signNMRU/errors/${ranges[i]}-${seed}.err & # parallel version

############################################## Hard #########################################################
    # op: div -> a / b where a and b are drawn from small uniform ranges

    # Real NPU
#    experiment_name='in2_divBy0/hard/realnpu'
#    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
#    --operation div --layer-type RealNPU --nac-mul real-npu \
#    --regualizer-shape none --regualizer 0 --learning-rate 5e-3 \
#    --regualizer-l1 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#    --npu-clip wg --npu-Wr-init xavier-uniform-constrained \
#    --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#    --regualizer-scaling-start 40000 --regualizer-scaling-end 50000 \
#    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/in2_divBy0/hard/realnpu/${ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/in2_divBy0/hard/realnpu/errors/${ranges[i]}-${seed}.err & # parallel version

    # NRU
#    experiment_name='in2_divBy0/hard/nru'
#    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
#    --operation div --layer-type NRU --nac-mul mnac \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1 \
#    > /data/nalms/logs/in2_divBy0/hard/nru/${ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/in2_divBy0/hard/nru/errors/${ranges[i]}-${seed}.err & # parallel version


    # NMRU
#    experiment_name='in2_divBy0/hard/signNMRU'
#    python3 -u /home/bm4g15/nalu-stable-exp/experiments/single_layer.py \
#    --operation div --layer-type SignNMRU \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${ranges[i]} --extrapolation-range ${ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda --no-save \
#    --clip-grad-norm 1 --learning-rate 1e-2 \
#    > /data/nalms/logs/in2_divBy0/hard/signNMRU/${ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/in2_divBy0/hard/signNMRU/errors/${ranges[i]}-${seed}.err & # parallel version

##############################################################################################################

  done
  wait
done
wait
date
echo "Script finished."
