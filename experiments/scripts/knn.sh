#!/usr/bin/env bash
#
#SBATCH --time=00:30:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started KNN classification"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%% OPTIM %%%%%%%%%%%%%%%%%%"
# python knn_classification.py --dataset optim --model mae-small --weights MAE_SSL_JUMP


echo "%%%%%%%%%%%%%%%%%%%%%%%%%%% Synaptic Proteins %%%%%%%%%%%%%%%%%%"
python knn_classification.py --dataset synaptic-proteins --model mae-small --weights MAE_SSL_JUMP
python knn_classification.py --dataset synaptic-proteins --model mae-base --weights MAE_BASE_SSL_STED

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
