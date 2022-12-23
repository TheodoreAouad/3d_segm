#!/bin/bash
#SBATCH --job-name=bisel_fashionmnist_erodila_75
#SBATCH --output=ruche_logs/bisel_fashionmnist_erodila_75.txt
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

python deep_morpho/train_net_segm.py --args deep_morpho/saved_args/bisel/fashionmnist/erosion_dilation/args_segm.py

wait
echo 'python scripts have finished'