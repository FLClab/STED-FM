#!/usr/bin/env bash
#
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_inter 
#SBATCH --array=0-24


#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    "MAE_TINY_IMAGENET1K_V1"
    "MAE_TINY_HPA"
    "MAE_TINY_JUMP"
    "MAE_TINY_SIM"
    "MAE_TINY_STED"
)

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
for weight in "${WEIGHTS[@]}"
do
    for dataset in "${DATASETS[@]}"
    do
        for seed in "${SEEDS[@]}"
        do
            opts+=("$weight;$dataset;$seed")
        done
    done
done

# # # Reads a specific item in the array and asign the values
# # # to the opt variable
IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
weight="${opt[0]}"
dataset="${opt[1]}"
seed="${opt[2]}"

cd evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started linear probing"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset $dataset --model mae-lightning-small --weights $weight --blocks "all" --seed ${seed} 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
