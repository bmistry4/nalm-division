#!/bin/bash
#SBATCH --partition=serial
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bm4g15@soton.ac.uk
#SBATCH --array=0-224                    # 0 to (seeds * |operations|=1 * |ranges|)-1  # seeds=25, range=9
#SBATCH --output /data/nalms/logs/sltr-in2/mixed-sign-ds_realnpu-modified/slurm-%A_%a.out # TODO: Make sure subfolder exists and matches experiment name.
#SBATCH --error /data/nalms/logs/sltr-in2/mixed-sign-ds_realnpu-modified/errors/slurm-%A_%a.err # TODO: Make subfolder exists and matches experiment name.

experiment_name='sltr-in2/mixed-sign-ds_realnpu-modified'
verbose_flag=''
no_save_flag='--no-save'

interpolation_ranges=( '[-2,2]'           '[[-2,-0.1],[0.1,2],[]]' '[[0.1,2],[-2,-0.1],[]]' '[[-2,-1],[1,2],[]]'  '[[1,2],[-2,-1],[]]' )
extrapolation_ranges=( '[[-6,-2],[2,6]]'  '[[-6,-2],[2,6],[]]'     '[[2,6],[-6,-2],[]]'     '[[-6,-2],[2,6],[]]'  '[[2,6],[-6,-2],[]]' )

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

# NAME: clip-WG_M-S40K-E50K-G1-W1_WrI-xuc
# RealNPU with the various modifications (relnpu mod)
python3 experiments/single_layer.py \
    --operation div --layer-type RealNPU --nac-mul real-npu \
    --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
    --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
    --interpolation-range ${interp} --extrapolation-range ${extrap} \
    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
    --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
    --regualizer-scaling-start 40000 --regualizer-scaling-end 50000 \
    --npu-Wr-init xavier-uniform-constrained