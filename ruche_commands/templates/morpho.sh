#!/bin/bash
#SBATCH --job-name={DATASET}_{OPERATION}_75
#SBATCH --output=ruche_logs/75_{DATASET}_{OPERATION}.txt
#SBATCH --mail-user=theodore.aouad@centralesupelec.fr
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:0:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --export=NONE

module purge
module list

cd $WORKDIR/3d_segm
pwd

echo $CUDA_VISIBLE_DEVICES
nproc

source $WORKDIR/virtual_envs/torchenv/bin/activate
echo 'Virtual environment activated'

python deep_morpho/train.py --args deep_morpho/saved_args/recompute_projected/{DATASET}/{OPERATION}/args.py

wait
echo 'python scripts have finished'