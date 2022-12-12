#!/bin/bash
#SBATCH --job-name=test_1
#SBATCH --output=logs_test_1.txt
#SBATCH --mail-user=theodore.aouad@centralesupelec.fr
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --export=NONE

module purge
module list

cd $WORKDIR/test_1
pwd

echo $CUDA_VISIBLE_DEVICES

source activate env_pytorch_geometric_vs_02
echo 'Virtual environment activated'
python random_search_hyperparameters_deeper.py --dataset Actor --gpu_number 0 --n_layers_set 32 --RicciCurvature > hyperpar_tuning_deep_txt/RC_Actor_nl_32_run_03.txt &
python random_search_hyperparameters_deeper.py --dataset Citeseer --gpu_number 0 --n_layers_set 2 --RicciCurvature > hyperpar_tuning_deep_txt/RC_Citeseer_nl_2_run_02.txt &
wait
echo 'python scripts have finished'