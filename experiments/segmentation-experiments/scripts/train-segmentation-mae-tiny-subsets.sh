#!/usr/bin/env bash
#
#SBATCH --time=08:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-224
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
    "freeze_backbone true batch_size 32"
    "freeze_backbone false batch_size 32"
    "freeze_backbone true batch_size 32"
    "freeze_backbone false batch_size 32"
    "freeze_backbone true batch_size 32"
    "freeze_backbone false batch_size 32"        
    "freeze_backbone true batch_size 32"
    "freeze_backbone false batch_size 32"
    "freeze_backbone false batch_size 32"
)
SUBSETS=(
    10
    25
    50
    100
    250
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
    for subset in "${SUBSETS[@]}"
    do
        for i in $(seq 0 8)
        do
            opts+=("${seed};$subset;${BACKBONEWEIGHTS[$i]};${OPTS[$i]}")
        done
    done
done
# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
seed="${opt[0]}"
subset="${opt[1]}"
weight="${opt[2]}"
options="${opt[3]}"

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/segmentation-experiments

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "% Seed: ${seed}"
echo "% Subset: ${subset}"
echo "% Weight: ${weight}"
echo "% Options: ${options}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

tensorboard --logdir="/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/segmentation-experiments/logs" --host 0.0.0.0 --load_fast false &
python main.py --seed ${seed} --use-tensorboard --dataset "factin" \
    --backbone "mae-lightning-small" --backbone-weights ${weight} \
    --num-per-class ${subset} --opts ${options}

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"