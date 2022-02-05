#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bm4g15@soton.ac.uk
#SBATCH --array=0-224                    #  0 to (seeds * |operations|=1 * |ranges|)-1  # seeds=25, range=9
#SBATCH --output /data/nalms/logs/sltr-in2/realnpu-modifications/L1_beta_sweep/1e-8_1e-6/slurm-%A_%a.out # TODO: Make sure folder path exists and matches exp name. (Same for err dir).
#SBATCH --error /data/nalms/logs/sltr-in2/realnpu-modifications/L1_beta_sweep/1e-8_1e-6/errors/slurm-%A_%a.err # TODO: Make sure folder path exists and matches exp name. (Same for err dir).

experiment_name='sltr-in2/realnpu-modifications/L1_beta_sweep/1e-8_1e-6' # TODO: UPDATE NAME
verbose_flag=''
no_save_flag='--no-save'

# TODO- update manually: (start,end): (1e-11,1e-9) (1e-8,1e-6) (1e-9,1e-7) (1e-7,1e-5) (1e-5,1e-3) (1e-3,1e-1) (1e-1,10)
beta_start=1e-8
beta_end=1e-6

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

# Baseline RealNPU with l1 reg
python3 experiments/single_layer.py \
    --operation div --layer-type RealNPU --nac-mul real-npu \
    --learning-rate 5e-3 --regualizer-beta-start ${beta_start} --regualizer-beta-end ${beta_end} \
    --regualizer-l1 --regualizer-shape none --regualizer 0 \
    --interpolation-range ${interp} --extrapolation-range ${extrap} \
    --seed ${seed} --max-iterations 50000 ${verbose_flag} \
    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag}



