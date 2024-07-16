#!/usr/bin/env bash
#
#SBATCH --time=04:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

# Training options
RESTOREFROM=(
    "from-scratch"
    "pretrained-MAE_SMALL_IMAGENET1K_V1"
    "pretrained-frozen-MAE_SMALL_IMAGENET1K_V1"
    "pretrained-MAE_SMALL_SSL_STED"
    "pretrained-frozen-MAE_SMALL_SSL_STED"
)

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/segmentation-experiments

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started evaluation"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python eval.py \
    --restore-from "./data/SSL/segmentation-baselines/mae-small/footprocess/${RESTOREFROM[${SLURM_ARRAY_TASK_ID}]}/result.pt" \
    --dataset "footprocess" \
    --opts batch_size 32

python eval-per-percentage.py \
    --restore-from "./data/SSL/segmentation-baselines/mae-small/footprocess/${RESTOREFROM[${SLURM_ARRAY_TASK_ID}]}/" \
    --dataset "footprocess" \
    --opts batch_size 32    

python eval-per-epochs.py \
    --restore-from "./data/SSL/segmentation-baselines/mae-small/footprocess/${RESTOREFROM[${SLURM_ARRAY_TASK_ID}]}/" \
    --dataset "footprocess" \
    --opts batch_size 32        

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done evaluation"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"