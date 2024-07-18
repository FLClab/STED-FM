#!/usr/bin/env bash
#
#SBATCH --time=0:59:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODELS=(
    "supervised-mae-lightning-tiny"
    "supervised-mae-lightning-small"
)



model=${MODELS[${SLURM_ARRAY_TASK_ID}]}

cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started evaluation on the test set"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python eval_v2.py --dataset neural-activity-states --model $model
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
