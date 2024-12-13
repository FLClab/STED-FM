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
#SBATCH --array=0-24


#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# WEIGHTS=(
#     "MAE_TINY_IMAGENET1K_V1"
#     "MAE_TINY_HPA"
#     "MAE_TINY_JUMP"
#     "MAE_TINY_SIM"
#     "MAE_TINY_STED"
# )

DATASETS=(
    "optim"
    "neural-activity-states"
    "peroxisome"
    "polymer-rings"
    "dl-sim"
)

SEEDS=(
    42
    43
    44
    45
    46
)

opts=()
for dataset in "${DATASETS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        opts+=("$dataset;$seed")
    done
done

# # # Reads a specific item in the array and asign the values
# # # to the opt variable
IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
dataset="${opt[0]}"
seed="${opt[1]}"

# weight=${WEIGHTS[${SLURM_ARRAY_TASK_ID}]}
# dataset=${DATASETS[${SLURM_ARRAY_TASK_ID}]}
# seed=${SEEDS[${SLURM_ARRAY_TASK_ID}]}



cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started linear probing"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset $dataset --model mae-lightning-tiny --weights "MAE_TINY_SIM" --blocks "all" --seed ${seed}
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"