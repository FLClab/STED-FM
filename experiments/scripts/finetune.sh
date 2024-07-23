#!/usr/bin/env bash
#
#SBATCH --time=8:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-2

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    "MAE_TINY_IMAGENET1K_V1"
    "MAE_TINY_JUMP"
    "MAE_TINY_STED"
)

weight=${WEIGHTS[${SLURM_ARRAY_TASK_ID}]}


cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started linear probing"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights $weight --blocks "0" 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
