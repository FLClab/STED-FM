#!/usr/bin/env bash
#
#SBATCH --time=03:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=koles2@ulaval.ca
#SBATCH --mail-type=ALL
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/myenv
source $VENV_DIR/bin/activate

# Training options
BACKBONEWEIGHTS=(
    # "MAE_TINY_HPA"
    # "MAE_TINY_HPA"
    "MAE_SMALL_IMAGENET1K_V1"
    "MAE_SMALL_IMAGENET1K_V1"
    # "MAE_TINY_STED"
    # "MAE_TINY_STED"
    # None
)
OPTS=(
    # "freeze_backbone true"
    # "freeze_backbone false"
    "freeze_backbone true"
    "freeze_backbone false"
#     "freeze_backbone true"
#     "freeze_backbone false"
#     "freeze_backbone false"
)

# Moves to working directory
cd ${HOME}/flc-dataset/experiments/segmentation-experiments

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# tensorboard --logdir="/scratch/anbil106/anbil106/SSL/segmentation-baselines" --host 0.0.0.0 --load_fast false &
tensorboard --logdir="./data/SSL/segmentation-baselines" --host 0.0.0.0 --load_fast false &
python main.py --seed 42 --use-tensorboard --dataset "factin" \
    --backbone "mae-lightning-small" --backbone-weights ${BACKBONEWEIGHTS[${SLURM_ARRAY_TASK_ID}]} \
    --opts ${OPTS[${SLURM_ARRAY_TASK_ID}]}

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"