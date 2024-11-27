#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-119

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/evaluation

WEIGHTS=(
    "RESNET50_IMAGENET1K_V1"
    "RESNET50_SSL_HPA"
    "RESNET50_SSL_JUMP"
    "RESNET50_SSL_STED"
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

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started fine tuning in low data regime ($numclass samples per class)"
echo $weight
echo $numclass 
echo $seed
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset optim --model resnet50 --weights $weight --blocks "all" --num-per-class $numclass --seed $seed
python finetune_v2.py --dataset neural-activity-states --model resnet50 --weights $weight --blocks "all" --num-per-class $numclass --seed $seed
python finetune_v2.py --dataset polymer-rings --model resnet50 --weights $weight --blocks "all" --num-per-class $numclass --seed $seed
python finetune_v2.py --dataset peroxisome --model resnet50 --weights $weight --blocks "all" --num-per-class $numclass --seed $seed

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
