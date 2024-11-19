#!/usr/bin/env bash
#
#SBATCH --time=1:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=4
#SBATCH --mem=100Gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp "/home/frbea320/projects/def-flavielc/datasets/FLCDataset/dataset-250k.tar" "${SLURM_TMPDIR}/dataset-250k.tar"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Beginning..."
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python build_user_study.py --seed 42 --dataset-path "${SLURM_TMPDIR}/dataset-250k.tar"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"