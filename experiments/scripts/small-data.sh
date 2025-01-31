#!/usr/bin/env bash
#
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gres=shard:2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-99
#SBATCH --partition=gpu_inter

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    "MAE_SMALL_IMAGENET1K_V1"
    "MAE_SMALL_JUMP"
    "MAE_SMALL_HPA"
    "MAE_SMALL_SIM"
    "MAE_SMALL_STED"
)

NUMCLASSES=(
    10
    25
    50
    100
)

SEEDS=(
    42
    43
    44
    45
    46
)

params=()
# for weight in "${WEIGHTS[@]}"
# do
for numclass in "${NUMCLASSES[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        for weight in "${WEIGHTS[@]}"
        do
            params+=("$numclass;$seed;$weight")
        done
    done
done
# done

# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a param <<< "${params[${SLURM_ARRAY_TASK_ID}]}"
numclass="${param[0]}"
seed="${param[1]}"
weight="${param[2]}"

cd evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started fine tuning in low data regime ($numclass samples per class)"
echo $numclass 
echo $seed
echo $weight
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights $weight --blocks "all" --num-per-class $numclass --seed $seed

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"