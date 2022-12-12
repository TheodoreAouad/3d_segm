#!/bin/bash
#SBATCH --job-name=test_1
#SBATCH --output=logs_test_1.txt
#SBATCH --mail-user=theodore.aouad@centralesupelec.fr
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --export=NONE

module purge
module list

cd $WORKDIR/3d_segm
pwd

echo $CUDA_VISIBLE_DEVICES

source $WORKDIR/virtual_envs/torchenv/bin/activate
echo 'Virtual environment activated'

python deep_morpho/train_net_segm.py

wait
echo 'python scripts have finished'