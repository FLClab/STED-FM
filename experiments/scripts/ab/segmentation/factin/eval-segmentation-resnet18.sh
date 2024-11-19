#!/usr/bin/env bash
#
#SBATCH --time=04:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --array=0
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

# Training options
MODEL="resnet18"
DATASET="factin"
SEEDS=(
    42
    43
    44
    45
    46
)
RESTOREFROM=(
    "from-scratch"
    "pretrained-RESNET18_IMAGENET1K_V1"
    "pretrained-frozen-RESNET18_IMAGENET1K_V1"
    "pretrained-RESNET18_SSL_JUMP"
    "pretrained-frozen-RESNET18_SSL_JUMP"
    "pretrained-RESNET18_SSL_HPA"
    "pretrained-frozen-RESNET18_SSL_HPA"
    "pretrained-RESNET18_SSL_STED"
    "pretrained-frozen-RESNET18_SSL_STED"
)

opts=()
for seed in "${SEEDS[@]}"
do
    for restorefrom in "${RESTOREFROM[@]}"
    do
        opts+=("$seed;$restorefrom")
    done
done
# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
seed="${opt[0]}"
restorefrom="${opt[1]}"

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/segmentation-experiments
BASE_PATH="/home/anbil106/scratch/projects/SSL"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started evaluation"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python eval.py \
    --restore-from "${BASE_PATH}/segmentation-baselines/${MODEL}/${DATASET}/${restorefrom}-${seed}/result.pt" \
    --dataset "${DATASET}" \
    --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from "${BASE_PATH}/segmentation-baselines/${MODEL}/factin/${RESTOREFROM[${SLURM_ARRAY_TASK_ID}]}/" \
#     --dataset "factin" \
#     --opts batch_size 32    

# python eval-per-epochs.py \
#     --restore-from "${BASE_PATH}/segmentation-baselines/${MODEL}/factin/${RESTOREFROM[${SLURM_ARRAY_TASK_ID}]}/" \
#     --dataset "factin" \
#     --opts batch_size 32        

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done evaluation"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"