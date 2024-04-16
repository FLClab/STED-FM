#!/usr/bin/env bash
#
#SBATCH --time=01:00:00
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

# Moves to working directory
cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

# WEIGHTS=(
#     "ImageNet"
#     "CTC"
#     "STED"
# )

TASKS=(
    "linear-probe"
    "finetuned"
)

# opts=()
# for weight in "${WEIGHTS[@]}"
# do 
#     for task in "${TASKS[@]}"
#     do
#         opts+=("$weight, $task")
#     done 
# done 

task=${TASKS[${SLURM_ARRAY_TASK_ID}]}

# IFS=', ' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
# weight="${opt[0]}"
# task="${opt[1]}"


# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started fine-tuning $task"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python eval_per_epoch.py --dataset synaptic-proteins --model MAE --task $task

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
