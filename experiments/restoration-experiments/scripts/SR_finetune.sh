#!/bin/bash 

#SBATCH --time=12:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export TORCH_NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/links/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


cd ${HOME}/links/projects/def-flavielc/frbea320/STED-FM/experiments/restoration-experiments


echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Beginning..."
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

tensorboard --logdir="/home/frbea320/links/scratch/denoising-baselines" --host 0.0.0.0 --load_fast false &
python main.py --use-tensorboard --dataset "dendritic-factin" --backbone "mae-lightning-small" --backbone-weights "MAE_SMALL_STED" --opts "freeze_backbone false full_decoder true" --save-folder "/home/frbea320/links/scratch/SR-baselines"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
