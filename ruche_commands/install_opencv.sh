#!/bin/bash
#SBATCH --job-name=test_1
#SBATCH --output=logs_test_1.txt
#SBATCH --mail-user=theodore.aouad@centralesupelec.fr
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_test
#SBATCH --export=NONE

module purge
module list

module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

cd $WORKDIR/test_1
pwd

echo $CUDA_VISIBLE_DEVICES

source activate torchenv
echo 'Virtual environment activated'
conda install -c "conda-forge/label/gcc7" opencv
wait
echo 'python scripts have finished'