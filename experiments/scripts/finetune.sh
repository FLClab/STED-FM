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
#SBATCH --array=0-99

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    "MAE_SMALL_JUMP"
    "MAE_SMALL_HPA"
    "MAE_SMALL_STED"
    "MAE_SMALL_HYBRID"
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
for dataset in "${DATASETS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        for weight in "${WEIGHTS[@]}"
        do
            opts+=("$dataset;$seed;$weight")
        done
    done
done

IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
dataset="${opt[0]}"
seed="${opt[1]}"
weight="${opt[2]}"


cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started fine tuning"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset $dataset --model mae-lightning-small --weights $weight --blocks "0" --seed ${seed} --overwrite
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
