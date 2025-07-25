#!/usr/bin/env bash
#
#SBATCH --time=48:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --nodes=2
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

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp "/home/frbea320/projects/def-flavielc/datasets/FLCDataset/dataset-250k.tar" "${SLURM_TMPDIR}/dataset-250k.tar"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Beginning..."
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# python main.py --dataset-path "${SLURM_TMPDIR}/dataset-250k.tar"
tensorboard --logdir="./model-checkpoints" --host 0.0.0.0 --load_fast false &
srun python train_classifier_guidance.py --seed 42 --dataset-path "${SLURM_TMPDIR}/dataset-250k.tar" --use-tensorboard

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
