#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-14%1

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
# cd ${HOME}/Documents/flc-dataset/experiments/simclr-experiments


cp "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/JUMP_CP/jump.hdf5" "${SLURM_TMPDIR}/jump.hdf5"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started pre-training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

if test -e /home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/JUMP_CP/baselines/mae-small/current_model.pth; then 
	python pretrain.py --seed 42 --modality "JUMP" --model "mae-small" --dataset-path "${SLURM_TMPDIR}/jump.hdf5" --restore-from "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/JUMP_CP/baselines/mae-small/current_model.pth" --save-folder "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/JUMP_CP/baselines/mae-small/"
else 
	python pretrain.py --seed 42 --modality "JUMP" --model "mae-small" --dataset-path "${SLURM_TMPDIR}/jump.hdf5" --save-folder "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/JUMP_CP/baselines/mae-small/"
fi

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
