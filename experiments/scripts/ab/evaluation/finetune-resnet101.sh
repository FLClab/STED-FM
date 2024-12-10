#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEEDS=(
    42
    43
    44
    45
    46
)
DATASETS=(
    "optim"
    "neural-activity-states"
    "peroxisome"
    "polymer-rings"
)

seed="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
params=()
for dataset in "${DATASETS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        params+=("$dataset;$seed")
    done
done

# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a param <<< "${params[${SLURM_ARRAY_TASK_ID}]}"
dataset="${param[0]}"
seed="${param[1]}"

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started finetuning"
echo "% dataset: ${dataset}"
echo "% seed: ${seed}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_IMAGENET1K_V1 --blocks 0 --seed $seed --overwrite
python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SSL_HPA --blocks 0 --seed $seed --overwrite
python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SSL_JUMP --blocks 0 --seed $seed --overwrite
python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SSL_SIM --blocks 0 --seed $seed --overwrite
python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SSL_STED --blocks 0 --seed $seed --overwrite

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"