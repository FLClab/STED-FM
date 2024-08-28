#!/usr/bin/env bash
#
#SBATCH --time=01:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-6
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
    "MAE_SMALL_HPA"
    "MAE_SMALL_HPA"
    "MAE_SMALL_IMAGENET1K_V1"
    "MAE_SMALL_IMAGENET1K_V1"
    "MAE_SMALL_STED"
    "MAE_SMALL_STED"
    None
)
OPTS=(
    "freeze_backbone true"
    "freeze_backbone false"
    "freeze_backbone true"
    "freeze_backbone false"
    "freeze_backbone true"
    "freeze_backbone false"
    "freeze_backbone false"
)

# Moves to working directory
cd ${HOME}/flc-dataset/experiments/segmentation-experiments

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# tensorboard --logdir="/scratch/anbil106/anbil106/SSL/segmentation-baselines" --host 0.0.0.0 --load_fast false &
tensorboard --logdir="/home/koles2/scratch/ssl_project/segmentation_baselines_test" --port=6007 --host 0.0.0.0 --load_fast false &
python main.py --seed 42 --save-folder "/home/koles2/scratch/ssl_project/segmentation_baselines_test0" --use-tensorboard --dataset "footprocess" \
    --backbone "mae-lightning-small" --backbone-weights ${BACKBONEWEIGHTS[${SLURM_ARRAY_TASK_ID}]} \
    --opts ${OPTS[${SLURM_ARRAY_TASK_ID}]} \
    --num-samples 5

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"