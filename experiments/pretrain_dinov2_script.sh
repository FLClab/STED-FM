#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gb
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export TORCH_NCCL_BLOCKING_WAIT=1

module load python/3.12 scipy-stack
module load cuda/12.6 cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#### PARAMETERS — À MODIFIER
MODEL="dinov2-lightning-small"
DATASET="STED"
DATASET_PATH="/path/to/stedfm-dataset-crops.tar"   # <-- mettre le vrai chemin ici
SAVE_FOLDER="/home/frbea320/scratch/baselines/dinov2-small_STED"
SEED=42
####

cd /home/frbea320/projects/def-flavielc/frbea320/STED-FM

mkdir -p $SAVE_FOLDER
mkdir -p logs

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DINOv2 Pretraining"
echo "% Model   : $MODEL"
echo "% Dataset : $DATASET ($DATASET_PATH)"
echo "% Output  : $SAVE_FOLDER"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

srun python experiments/pretrain_dinov2_lightning.py \
    --seed $SEED \
    --model $MODEL \
    --dataset $DATASET \
    --dataset-path "$DATASET_PATH" \
    --save-folder "$SAVE_FOLDER" \
    --use-tensorboard

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
