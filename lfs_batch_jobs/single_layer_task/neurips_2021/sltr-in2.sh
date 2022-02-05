#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bm4g15@soton.ac.uk
#SBATCH --array=0-224                    # 0 to (seeds * |operations|=1 * |ranges|)-1  # seeds=25, range=9
#SBATCH --output /data/nalms/logs/sltr-in2/realnpu_baseline/slurm-%A_%a.out # TODO: Make sure folder path exists and matches exp name. (Same for err dir).
#SBATCH --error /data/nalms/logs/sltr-in2/realnpu_baseline/errors/slurm-%A_%a.err

verbose_flag=''
no_save_flag='--no-save'

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

export TENSORBOARD_DIR=/data/nalms/tensorboard
export SAVE_DIR=/data/nalms/saves
export PYTHONPATH=./

# TODO - uncomment relevant model and run
# RealNPU (baseline)
experiment_name='sltr-in2/realnpu_baseline'
python3 experiments/single_layer.py \
  --operation div --layer-type RealNPU --nac-mul real-npu \
  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
  --regualizer-l1 --regualizer-shape none --regualizer 0 \
  --interpolation-range ${interp} --extrapolation-range ${extrap} \
  --seed ${seed} --max-iterations 50000 ${verbose_flag} \
  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag}

# RealNPU
# NAME: WG_clip-M-S40000-E50000-G1-W1-WrI_xuc
# RealNPU -> l1 reg & WG clipping & G1 and W1 reg + NAU init for W_real
#experiment_name='sltr-in2/realnpu_modified'
#python3 experiments/single_layer.py \
#    --operation div --layer-type RealNPU --nac-mul real-npu \
#    --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#    --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
#    --interpolation-range ${interp} --extrapolation-range ${extrap} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} ${no_save_flag} \
#    --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#    --regualizer-scaling-start 40000 --regualizer-scaling-end 50000 \
#    --npu-Wr-init xavier-uniform-constrained

# NRU
#experiment_name='SLTR_NRU/identity-conversion-approx-1000-tanh1000_lr-1'
#experiment_name='sltr-in2/nru'
#python3 experiments/single_layer.py \
#    --operation div --layer-type NRU --nac-mul mnac \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1


# NMRU
#experiment_name='sltr-in2/sign-nmru'
#python3 experiments/single_layer.py \
#    --operation div --layer-type SignNMRU \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    --clip-grad-norm 1 --learning-rate 1e-2

############################################## Real NPU (modifications) ###############################################
## L1 regularisation --> off = Baseline RealNPU WITHOUT l1 reg or w&g reg
#experiment_name='sltr-in2/realnpu-modifications/L1F'
#python3 experiments/single_layer.py \
#    --operation div --layer-type RealNPU --nac-mul real-npu \
#    --learning-rate 5e-3 \
#    --regualizer-shape none --regualizer 0 \
#    --interpolation-range ${interp} --extrapolation-range ${extrap} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag}

# (on) --> see run for RealNPU (baseline)

## clipping -> 'None', 'G', 'W', 'GW'
# (None) -> see run for RealNPU (baseline)

#experiment_name='sltr-in2/realnpu-modifications/clip/G'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --npu-clip g

#experiment_name='sltr-in2/realnpu-modifications/clip/W'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --npu-clip w

#experiment_name='sltr-in2/realnpu-modifications/clip/GW'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --npu-clip wg

## discretisation -> 'None','G', 'GW'
# (None) -> see run for GW clip above

#experiment_name='sltr-in2/realnpu-modifications/discretisation/G1'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --npu-clip wg --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 40000 --regualizer-scaling-end 50000 \

#experiment_name='sltr-in2/realnpu-modifications/discretisation/G1-W1'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --npu-clip wg --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 40000 --regualizer-scaling-end 50000 \

## initalisation -> 'Xavier-Uniform', 'Xavier-Uniform Constrained'
# Xavier-Uniform -> (see above WG discretisation)
#experiment_name='sltr-in2/realnpu-modifications/init/XUC'
#python3 experiments/single_layer.py \
#  --operation div --layer-type RealNPU --nac-mul real-npu \
#  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
#  --regualizer-l1 --regualizer-shape none --regualizer 0 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --npu-clip wg --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
#  --regualizer-scaling-start 40000 --regualizer-scaling-end 50000 \
#  --npu-Wr-init xavier-uniform-constrained

# (Real NPU - mixed-signs datasets) -- see separate shell

# (Different distributions) -- see separate shell


############################################## Appendix #########################################################
## NRU learning rates
# lr: 1e-3
#experiment_name='sltr-in2/nru-extra/lr-1e-3'
#python3 experiments/single_layer.py \
#    --operation div --layer-type NRU --nac-mul mnac \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    --learning-rate 1e-3

# lr: 1e-2
#experiment_name='sltr-in2/nru-extra/lr-1e-2'
#python3 experiments/single_layer.py \
#    --operation div --layer-type NRU --nac-mul mnac \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    --learning-rate 1e-2

# lr: 1e-1
#experiment_name='sltr-in2/nru-extra/lr-1e-1'
#python3 experiments/single_layer.py \
#    --operation div --layer-type NRU --nac-mul mnac \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    --learning-rate 1e-1

# realnpu reg -> l2 on
#experiment_name='realnpu-modifications/L2'
#python3 experiments/single_layer.py \
#    --operation div --layer-type RealNPU --nac-mul real-npu \
#    --learning-rate 5e-3 --regualizer-beta-start 1e-7 --regualizer-beta-end 1e-5 \
#    --regualizer-l1 --regualizer-shape none --regualizer 0 --lp-norm 2 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag}  \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag}

