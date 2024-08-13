#!/usr/bin/env bash
#
#SBATCH --time=2:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    "MAE_LARGE_IMAGENET1K_V1"
    "MAE_LARGE_HPA"
    "MAE_LARGE_JUMP"
    "MAE_LARGE_STED"
)

# DATASETS=(
#     "optim"
#     "synaptic-proteins"
# )

# opts=()
# for weight in "${WEIGHTS[@]}"
# do
#     for dataset in "${DATASETS[@]}"
#     do
#         opts+=("$weight;$dataset")
#     done
# done

# # Reads a specific item in the array and asign the values
# # to the opt variable
# IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
# weight="${opt[0]}"
# dataset="${opt[1]}"

weight=${WEIGHTS[${SLURM_ARRAY_TASK_ID}]}


cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started linear probing"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset optim --model mae-lightning-large --weights $weight --blocks "all"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
