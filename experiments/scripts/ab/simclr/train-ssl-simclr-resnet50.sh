#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --array=1-10%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/simclr-experiments

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

cp "/project/def-flavielc/datasets/FLCDataset/dataset.tar" "${SLURM_TMPDIR}/dataset.tar"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

tensorboard --logdir="/scratch/anbil106/anbil106/SSL/baselines" --host 0.0.0.0 --load_fast false &
python main.py --seed 42 --use-tensorboard --dataset-path "${SLURM_TMPDIR}/dataset.tar" --backbone "resnet50" \
    --restore-from "/scratch/anbil106/anbil106/SSL/baselines/resnet50/result.pt"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"