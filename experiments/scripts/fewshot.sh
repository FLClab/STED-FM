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
#SBATCH --array=0-11

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    "MAE_SMALL_IMAGENET"
    "MAE_SSL_CTC"
    "MAE_SSL_STED"
)

PERCENTAGES=(
    0.01
    0.1
    0.25
    0.5
)

opts=()
for weight in "${WEIGHTS[@]}"
do
    for perc in "${PERCENTAGES[@]}"
    do
        opts+=("$weight;$perc")
    done
done

# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
weight="${opt[0]}"
perc="${opt[1]}"

# Moves to working directory
cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started few-shot learning"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune.py --dataset synaptic-proteins --model mae-small --weights $weight --blocks "0" --label-percentage $perc

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
