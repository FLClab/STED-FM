#!/usr/bin/env bash
#
#SBATCH --time=12:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-2


#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

BLOCKS=(
    # "12"
    # "11"
    # "10"
    # "9"
    # "8"
    # "7"
    # "6"
    # "5"
    # "4"
    # "3"
    # "2"
    # "1"
    "0"
)

WEIGHTS=(
    "MAE_IMAGENET"
    "MAE_SSL_CTC"
    "MAE_SSL_STED"
)

opts=()
for block in "${BLOCKS[@]}"
do 
    for weight in "${WEIGHTS[@]}"
    do
        opts+=("$block, $weight")
    done 
done 

IFS=', ' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
block="${opt[0]}"
weight="${opt[1]}"


# block=${BLOCKS[${SLURM_ARRAY_TASK_ID}]}


# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started fine-tuning"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune.py --dataset synaptic-proteins --model mae --weights $weight --blocks $block

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
