#!/usr/bin/env bash
#
#SBATCH --time=48:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4  
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

export TORCH_NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp "/home/frbea320/scratch/datasets/FLCDataset/dataset-250k.tar" "${SLURM_TMPDIR}/dataset-250k.tar"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Beginning..."
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# python main.py --dataset-path "${SLURM_TMPDIR}/dataset-250k.tar"
tensorboard --logdir="/home/frbea320/scratch/model_checkpoints/DiffusionModels/latent-guidance" --host 0.0.0.0 --load_fast false &
srun python train_latent_guidance.py --seed 42 --dataset-path "${SLURM_TMPDIR}/dataset-250k.tar" --use-tensorboard --weights MAE_SMALL_IMAGENET1K_V1

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
