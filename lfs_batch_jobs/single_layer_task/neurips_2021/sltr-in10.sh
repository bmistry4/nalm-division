#!/bin/bash
#SBATCH --partition=serial
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bm4g15@soton.ac.uk
#SBATCH --array=0-224    # 0 to (seeds * |operations|=1 * |ranges|)-1  # seeds=25, range=9
#SBATCH --output /data/nalms/logs/sltr-in10/realnpu_baseline/slurm-%A_%a.out # TODO: Make sure folder path exists and matches exp name. (Same for err dir).
#SBATCH --error /data/nalms/logs/sltr-in10/realnpu_baseline/errors/slurm-%A_%a.err # TODO: Make sure folder path exists and matches exp name. (Same for err dir).

verbose_flag=''
no_save_flag='${no_save_flag}'

interpolation_ranges=( '[-20,-10]' '[-2,-1]' '[-1.2,-1.1]' '[-0.2,-0.1]' '[-2,2]'          '[0.1,0.2]' '[1,2]' '[1.1,1.2]' '[10,20]' )
extrapolation_ranges=( '[-40,-20]' '[-6,-2]' '[-6.1,-1.2]' '[-2,-0.2]'  '[[-6,-2],[2,6]]' '[0.2,2]'   '[2,6]' '[1.2,6]'  '[20,40]' )

seed=`expr $SLURM_ARRAY_TASK_ID \/ ${#interpolation_ranges[@]}`  # integer division, brackets require spacing and \

if [[ ${#interpolation_ranges[@]} > 1 ]]; then
	let range_idx="$SLURM_ARRAY_TASK_ID % ( ${#interpolation_ranges[@]} )"; else
	let range_idx=0
fi

interp=${interpolation_ranges[range_idx]}
extrap=${extrapolation_ranges[range_idx]}

module load conda/py3-latest
source deactivate
conda activate nalu-env
cd /home/bm4g15/nalu-stable-exp/

export TENSORBOARD_DIR=/scratch/bm4g15/data/nalu-stable-exp/tensorboard
export SAVE_DIR=/scratch/bm4g15/data/nalu-stable-exp/saves
export PYTHONPATH=./

# TODO - uncomment relevant model and run
# RealNPU -> baseline
experiment_name='sltr-in10/realnpu_baseline'
python3 experiments/single_layer.py \
  --operation div --layer-type RealNPU --nac-mul real-npu \
  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
  --regualizer-l1 --regualizer-shape none --regualizer 0 \
  --interpolation-range ${interp} --extrapolation-range ${extrap} \
  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag}

# RealNPU -> l1 reg & WG clipping & G1 and W1 reg + NAU init for W_real
#experiment_name='sltr-in10/realnpu_modified'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --npu-Wr-init xavier-uniform-constrained

# NRU
#experiment_name='sltr-in10/nru'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NRU --nac-mul mnac \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1e-3 \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0

# NMRU
#experiment_name='sltr-in10/sign-nmru'
#  python3 experiments/single_layer.py \
#    --operation div --layer-type SignNMRU \
#    --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-2

############################################## Appendix #########################################################
############################################## in=10 (with redundancy)
## NPU -> 'Real NPU (modified)', 'NPU (no constraints)', 'NPU (clip & reg)'
# Real NPU (modified) = see realnpu_modified above

# NPU (no constraints)
# relevant flags: --npu-clip wg
#experiment_name='sltr-in10/npu/realMod'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NPU --nac-mul npu \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --npu-Wr-init xavier-uniform-constrained

# NPU (clip & reg)
# relevant flags: --npu-clip wig
#experiment_name='sltr-in10/npu/realMod_Reg-Wim-l1_clip-wig'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NPU --nac-mul npu \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wig \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --npu-Wr-init xavier-uniform-constrained


## Real NPU - W_im reg {-1,0,1}
# relevant flags: --realnpu-reg-type bias
#experiment_name='sltr-in10/realnpu_modified_W-nau-reg'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --npu-Wr-init xavier-uniform-constrained \
#  --realnpu-reg-type bias


## NMRU - ablation
# vanilla - no gnc + no sign
# relevant flags: --layer-type NMRU
#experiment_name='sltr-in10/nmru-extra/nmru'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --learning-rate 1e-2

# gnc + no sign
# relevant flags: --layer-type NMRU --clip-grad-norm 1
#experiment_name='sltr-in10/nmru-extra/nmru_gnc-1'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --clip-grad-norm 1 --learning-rate 1e-2

# sign + no gnc
# relevant flags: --layer-type SignNMRU
#experiment_name='sltr-in10/nmru-extra/sign-nmru_gnc-F'
#python3 experiments/single_layer.py \
#  --operation div --layer-type SignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --learning-rate 1e-2

# (gnc+sign) - covered in exp name: sltr-in10/sign-nmru

# gnc+sign+gating
# relevant flags: --layer-type GatedSignNMRU --clip-grad-norm 1 --reg-scale-type madsen --regualizer-gate 1
#experiment_name='sltr-in10/nmru-extra/sign-nmru_gnc-1_gate-1'
#python3 experiments/single_layer.py \
#  --operation div --layer-type GatedSignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-2 \
#  --reg-scale-type madsen --regualizer-gate 1

# NMRU - learning rates
# relevant flags: --learning-rate 1e-3
#experiment_name='sltr-in10/nmru-extra/lr-1e-3'
#python3 experiments/single_layer.py \
#  --operation div --layer-type SignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-3

# relevant flags: --learning-rate 1e-1
#experiment_name='sltr-in10/nmru-extra/lr-1e-1'
#python3 experiments/single_layer.py \
#  --operation div --layer-type SignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-1

# NMRU - optimiser
# relevant flags: --optimizer sgd
#experiment_name='sltr-in10/nmru-extra/sgd'
#    python3 experiments/single_layer.py \
#  --operation div --layer-type SignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-2 \
#  --optimizer sgd

# NRU - separating sign and magnitude calculation
# relevant flags: --nru-div-mode div-sepSign
#experiment_name='sltr-in10/nru-sepSign'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NRU --nac-mul mnac \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1e-3 \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --nru-div-mode div-sepSign

###################### Losses
### Real NPU
# PCC
#experiment_name='sltr-in10/realnpu_modified_pcc'
#  python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --npu-Wr-init xavier-uniform-constrained --train-criterion pcc

# MAPE
#experiment_name='sltr-in10/realnpu_modified_mape'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --npu-Wr-init xavier-uniform-constrained --train-criterion mape

### NRU
# PCC
#experiment_name='sltr-in10/nru_pcc'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NRU --nac-mul mnac \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1e-3 \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --train-criterion pcc

# MAPE
#experiment_name='sltr-in10/nru_mape'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NRU --nac-mul mnac \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1e-3 \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --train-criterion mape

### NMRU
# PCC
#experiment_name='sltr-in10/sign-nmru_pcc'
#python3 experiments/single_layer.py \
#  --operation div --layer-type SignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-2  --train-criterion pcc

# MAPE
#experiment_name='sltr-in10/sign-nmru_mape'
#python3 experiments/single_layer.py \
#  --operation div --layer-type SignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-2  --train-criterion mape
