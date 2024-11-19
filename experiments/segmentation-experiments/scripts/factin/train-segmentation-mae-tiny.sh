#!/usr/bin/env bash
#
#SBATCH --time=08:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-45
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Training options
BACKBONEWEIGHTS=(
    "MAE_TINY_IMAGENET1K_V1"
    "MAE_TINY_IMAGENET1K_V1"
    "MAE_TINY_HPA"
    "MAE_TINY_HPA"    
    "MAE_TINY_JUMP"
    "MAE_TINY_JUMP"
    "MAE_TINY_STED"
    "MAE_TINY_STED"    
    "None"
)
OPTS=(
    "freeze_backbone true batch_size 64"
    "freeze_backbone false batch_size 64"
    "freeze_backbone true batch_size 64"
    "freeze_backbone false batch_size 64"
    "freeze_backbone true batch_size 64"
    "freeze_backbone false batch_size 64"
    "freeze_backbone true batch_size 64"
    "freeze_backbone false batch_size 64"
    "freeze_backbone false batch_size 64"
)
SEEDS=(
    42
    43
    44
    45
    46
)

opts=()
for seed in "${SEEDS[@]}"
do
    for i in $(seq 0 8)
    do
        opts+=("$seed;${BACKBONEWEIGHTS[$i]};${OPTS[$i]}")
    done
done
# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
seed="${opt[0]}"
weight="${opt[1]}"
options="${opt[2]}"


# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "% Seed: ${seed}"
echo "% Weight: ${weight}"
echo "% Options: ${options}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

tensorboard --logdir="/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/segmentation-experiments/logs" --host 0.0.0.0 --load_fast false &
python main.py --seed ${seed} --use-tensorboard --dataset "factin" \
    --backbone "mae-lightning-tiny" --backbone-weights ${weight} \
    --opts ${options}

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"