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
#SBATCH --array=0-15

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

DATASETS=(
    "optim"
    "neural-activity-states"
    "peroxisome"
    "polymer-rings"
)

params=()
for weight in "${WEIGHTS[@]}"
do
    for dataset in "${DATASETS[@]}"
    do
        params+=("$weight;$dataset")
    done
done

# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a param <<< "${params[${SLURM_ARRAY_TASK_ID}]}"
weight="${param[0]}"
dataset="${param[1]}"

opts="batch_size 32"



cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started fine tuning"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset $dataset --model mae-lightning-large --weights $weight --blocks "0" --opts $opts
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
