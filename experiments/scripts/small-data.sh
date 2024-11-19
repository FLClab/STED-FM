#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-119

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    "MAE_BASE_IMAGENET1K_V1"
    "MAE_BASE_HPA"
    "MAE_BASE_JUMP"
    "MAE_BASE_STED"
)

NUMCLASSES=(
    10
    25
    50
    100
    250
    500
)

SEEDS=(
    42
    43
    44
    45
    46
)

params=()
for weight in "${WEIGHTS[@]}"
do
    for numclass in "${NUMCLASSES[@]}"
    do
        for seed in "${SEEDS[@]}"
        do
            params+=("$weight;$numclass;$seed")
        done
    done
done

# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a param <<< "${params[${SLURM_ARRAY_TASK_ID}]}"
weight="${param[0]}"
numclass="${param[1]}"
seed="${param[2]}"

# opts="batch_size 32"



cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started fine tuning in low data regime ($numclass samples per class)"
echo $weight
echo $numclass 
echo $seed
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights $weight --blocks "all" --num-per-class $numclass --seed $seed

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
