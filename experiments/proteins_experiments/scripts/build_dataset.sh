#!/bin/bash
#SBATCH --time=2:30:00
#SBATCH --account=def-flavielc
#SBATCH --mail-user=frederic.beaupre.3@ulaval.ca
#SBATCH --cpus-per-task=4
#SBATCH --mem=8Gb
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-type=ALL

module load python/3.8 scipy-stack
module load cuda cudnn

VENV_DIR=${HOME}/projects/def-flavielc/frbea320/ad
source $VENV_DIR/bin/activate

cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/proteins_experiments/datasets

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python build_dataset.py

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

