#!/usr/bin/env bash
#
#SBATCH --time=3:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-23

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    "MAE_SMALL_IMAGENET1K_V1"
    "MAE_SMALL_HPA"
    "MAE_SMALL_JUMP"
    "MAE_SMALL_STED"
)

NUMCLASSES=(
    10
    25
    50
    100
    250
    500
)

params=()
for weight in "${WEIGHTS[@]}"
do
    for numclass in "${NUMCLASSES[@]}"
    do
        params+=("$weight;$numclass")
    done
done

# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a param <<< "${params[${SLURM_ARRAY_TASK_ID}]}"
weight="${param[0]}"
numclass="${param[1]}"

# opts="batch_size 32"



cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started fine tuning in low data regime ($numclass samples per class)"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights $weight --blocks "all" --num-per-class $numclass --seed 43
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
