#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --array=1-14%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
# cd ${HOME}/Documents/flc-dataset/experiments/simclr-experiments

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

cp "./Datasets/Cell-Tracking-Challenge/cell-tracking-challenge.hdf5" "${SLURM_TMPDIR}/cell-tracking-challenge.hdf5"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

if test -e ./Datasets/Cell-Tracking-Challenge/baselines/current_model.pth; then 
	python train_lightly_mae.py --seed 42 --save-folder "./Datasets/Cell-Tracking-Challenge/baselines" --modality CTC --dataset-path "${SLURM_TMPDIR}/cell-tracking-challenge.hdf5" --restore-from "./Datasets/Cell-Tracking-Challenge/baselines/current_model.pth"
else 
	python train_lightly_mae.py --seed 42 --save-folder "./Datasets/Cell-Tracking-Challenge/baselines" --modality CTC --dataset-path "${SLURM_TMPDIR}/cell-tracking-challenge.hdf5"
fi

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
