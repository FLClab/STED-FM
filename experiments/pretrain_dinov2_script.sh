#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=4
#SBATCH --mem=64Gb
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 #Faire attention à la fraction du GPU (cf doc ComputeCanada)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=marez10@ulaval.ca
#SBATCH --mail-type=ALL

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export TORCH_NCCL_BLOCKING_WAIT=1

module load python/3.11.4 scipy-stack # attention à la version 
module load cuda/12.6 cudnn
source /home/mathis/links/projects/def-flavielc/mathis/sted-env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#### PARAMETERS — À MODIFIER
MODEL="dinov2-lightning-small"
DATASET="STED"
DATASET_PATH="/home/mathis/links/projects/def-flavielc/shared/datasets/FLCDataset/dataset.tar"
SAVE_FOLDER="/home/mathis/links/scratch/baselines/dinov2-small_STED"
SEED=42
MAX_EPOCHS=1000
####

cd /home/mathis/links/projects/def-flavielc/mathis/STED-FM

mkdir -p $SAVE_FOLDER
mkdir -p logs

# Détecter un checkpoint existant pour reprendre l'entraînement
CHECKPOINT="$SAVE_FOLDER/current_model.pth"
RESTORE_ARG=""
if [ -f "$CHECKPOINT" ]; then
    echo "Checkpoint found: $CHECKPOINT — resuming training"
    RESTORE_ARG="--restore-from $CHECKPOINT"
fi

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
    --use-tensorboard \
    $RESTORE_ARG

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# Relancer automatiquement si l'entraînement n'est pas terminé (désactivé)

# Faire attention quand on recharge sur un checkpoint, il faut recharger le state, scheduler, etc

# if [ -f "$CHECKPOINT" ]; then
#     LAST_EPOCH=$(python -c "
# import torch
# try:
#     ck = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
#     print(ck.get('epoch', 0))
# except:
#     print(0)
# ")
#     echo "Last completed epoch: $LAST_EPOCH / $((MAX_EPOCHS - 1))"
#     if [ "$LAST_EPOCH" -lt $((MAX_EPOCHS - 1)) ]; then
#         echo "Training not complete, resubmitting..."
#         sbatch "$0"
#     else
#         echo "Training complete!"
#     fi
# fi