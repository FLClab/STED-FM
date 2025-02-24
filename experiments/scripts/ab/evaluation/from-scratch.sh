#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-4
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/evaluation

MODELS=(
    "resnet18"
    "resnet50"
    "resnet101"
)
OPTIONS=(
    "batch_size 64"
    "batch_size 64"
    "batch_size 64"
)
SEEDS=(
    42
    43
    44
    45
    46
)

# Creates an array of possible options to choose from
opts=()
for i in $(seq 0 0);
do
    model="${MODELS[$i]}"
    option="${OPTIONS[$i]}"
    for seed in "${SEEDS[@]}"
    do
        opts+=("$model;$option;$seed")
    done
done

# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
model="${opt[0]}"
option="${opt[1]}"
seed="${opt[2]}"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started finetuning"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

echo -e "==============================================="
echo -e "==================== OPTIM ===================="
echo -e "==============================================="

python finetune_v2.py --seed $seed --dataset optim --model $model --from-scratch

echo -e "================================================================"
echo -e "==================== Neural Activity States ===================="
echo -e "================================================================"

python finetune_v2.py --seed $seed --dataset neural-activity-states --model $model --from-scratch

echo -e "===================================================="
echo -e "==================== Peroxisome ===================="
echo -e "===================================================="

python finetune_v2.py --seed $seed --dataset peroxisome --model $model --from-scratch

echo -e "======================================================="
echo -e "==================== Polymer Rings ===================="
echo -e "======================================================="

python finetune_v2.py --seed $seed --dataset polymer-rings --model $model --from-scratch

echo -e "======================================================="
echo -e "======================= DL-SIM ========================"
echo -e "======================================================="

python finetune_v2.py --seed $seed --dataset dl-sim --model $model --from-scratch

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
