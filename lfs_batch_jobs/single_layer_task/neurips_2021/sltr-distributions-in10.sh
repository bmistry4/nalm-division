#!/bin/bash
#SBATCH --partition=serial
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bm4g15@soton.ac.uk
#SBATCH --array=0-149                    # TODO: 0 to (seeds * |operations|=1 * |ranges|)-1  # seeds=25, range=6
#SBATCH --output /data/nalms/logs/sltr-in10/distributions/realnpu/slurm-%A_%a.out # TODO: Make sure folder path exists and matches exp name. (Same for err dir).
#SBATCH --error /data/nalms/logs/sltr-in10/distributions/realnpu/errors/slurm-%A_%a.err

verbose_flag=''
no_save_flag='--no-save'

# Additional dists
distributions=(        'benford'    'truncated-normal -1 3 -10 3'  'truncated-normal 1 3 10 3'  'truncated-normal 0 1 10 1' 'uniform'               'uniform'                 )
interpolation_ranges=( '[10,100]'   '[-5,10]'                      '[-10,5]'                    '[-5,5]'                    '[-50,50]'              '[-100,100]'              )
extrapolation_ranges=( '[100,1000]' '[-15,-5]'                     '[5,15]'                     '[5,15]'                    '[[-100,-50],[50,100]]' '[[-200,-100],[100,200]]' )

seed=`expr $SLURM_ARRAY_TASK_ID \/ ${#interpolation_ranges[@]}`  # integer division, brackets require spacing and \

if [[ ${#interpolation_ranges[@]} > 1 ]]; then
	let range_idx="$SLURM_ARRAY_TASK_ID % ( ${#interpolation_ranges[@]} )"; else
	let range_idx=0
fi

interp=${interpolation_ranges[range_idx]}
extrap=${extrapolation_ranges[range_idx]}
distribution=${distributions[range_idx]}

module load conda/py3-latest
source deactivate
conda activate nalu-env
cd /home/bm4g15/nalu-stable-exp/

export TENSORBOARD_DIR=/scratch/bm4g15/data/nalu-stable-exp/tensorboard
export SAVE_DIR=/scratch/bm4g15/data/nalu-stable-exp/saves
export PYTHONPATH=./

#################################################################################################################
### Input size 10 experiments for runs on different distributions.

# TODO - uncomment relevant model and run
# NAME: neurips2021_rebuttal/RealNPU-mod_inSize10
experiment_name='sltr-in10/distributions/realnpu'
python3 experiments/single_layer.py \
  --operation div --layer-type RealNPU --nac-mul real-npu \
  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
  --learning-rate 5e-3 --regualizer-beta-start 1e-9 --regualizer-beta-end 1e-7 \
  --regualizer-l1 --regualizer-shape none --regualizer 0 --npu-clip wg \
  --interpolation-range ${interp} --extrapolation-range ${extrap} \
  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
  --regualizer-npu-w 1 --regualizer-gate 1 --reg-scale-type madsen \
  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
  --npu-Wr-init xavier-uniform-constrained \
  --sample-distribution ${distribution}

# NAME: 'neurips2021_rebuttal/inSize-10_signNMRU-gnc-1_lr-1e-2'
#experiment_name='sltr-in10/distributions/sign-nmru'
#python3 experiments/single_layer.py \
#  --operation div --layer-type SignNMRU \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  ---name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 --clip-grad-norm 1 --learning-rate 1e-2 \
#  --sample-distribution ${distribution}

# NAME: 'neurips2021_rebuttal/inSize-10_NRU_lr-1e-3'
#experiment_name='sltr-in10/distributions/nru'
#python3 experiments/single_layer.py \
#  --operation div --layer-type NRU --nac-mul mnac \
#  --regualizer-scaling-start 50000 --regualizer-scaling-end 75000 \
#  --interpolation-range ${interp} --extrapolation-range ${extrap} \
#  --seed ${seed} --max-iterations 100000 ${verbose_flag} \
#  --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#  --learning-rate 1e-3 \
#  --input-size 10 --subset-ratio 0.1 --overlap-ratio 0 \
#  --sample-distribution ${distribution}