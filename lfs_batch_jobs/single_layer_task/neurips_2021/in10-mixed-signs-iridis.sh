#!/bin/bash
#SBATCH --partition=serial
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bm4g15@soton.ac.uk
#SBATCH --array=0-124                    # TODO: 0 to (seeds * |operations|=1 * |ranges|)-1  # seeds=25, range=5
#SBATCH --output /scratch/bm4g15/data/nalu-stable-exp/logs/mixedSigns-in10/realnpu_mod/slurm-%A_%a.out # TODO: Make sure folder path exists and matches exp name. (Same for err dir).
#SBATCH --error /scratch/bm4g15/data/nalu-stable-exp/logs/mixedSigns-in10/realnpu_mod/errors/slurm-%A_%a.err

verbose_flag=''
no_save_flag=''

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

export TENSORBOARD_DIR=/scratch/bm4g15/data/nalu-stable-exp/tensorboard
export SAVE_DIR=/scratch/bm4g15/data/nalu-stable-exp/saves
export PYTHONPATH=./

# TODO: uncomment relevant model
# Run command: sbatch in10-mixed-signs.sh

# RealNPU -> l1 reg & WG clipping & G1 and W1 reg + NAU init for W_real
experiment_name='mixedSigns-in10/realnpu_mod'
python3 experiments/single_layer.py \
  --id 3 --operation div --layer-type RealNPU --nac-mul real-npu \
  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
  --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
  --interpolation-range ${interp} --extrapolation-range ${extrap} \
  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
  --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
  --npu-Wr-init xavier-uniform-constrained

# NRU
#experiment_name='mixedSigns-in10/nru'
#python3 experiments/single_layer.py \
#  --id 4 --operation div --layer-type NRU --nac-mul mnac \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} --learning-rate 1e-3 \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0

# NMRU
#experiment_name='mixedSigns-in10/nmru'
#python3 experiments/single_layer.py \
#  --id 5 --operation div --layer-type SignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-2

