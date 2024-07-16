#!/usr/bin/env bash
#
#SBATCH --time=04:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-19
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
BACKBONEWEIGHTS=(
    "RESNET50_IMAGENET1K_V1"
    "RESNET50_IMAGENET1K_V1"
    "RESNET50_SSL_STED"
    "RESNET50_SSL_STED"
    "None"
)
OPTS=(
    "freeze_backbone true batch_size 64"
    "freeze_backbone false batch_size 64"
    "freeze_backbone true batch_size 64"
    "freeze_backbone false batch_size 64"
    "freeze_backbone false batch_size 64"
)
SUBSETS=(
    "0.01"
    "0.10"
    "0.25"
    "0.50"
)

opts=()
for subset in "${SUBSETS[@]}"
do
    for i in $(seq 0 3)
    do
        opts+=("$subset;${BACKBONEWEIGHTS[$i]};${OPTS[$i]}")
    done
done
# Reads a specific item in the array and asign the values
# to the opt variable
IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
subset="${opt[0]}"
weight="${opt[1]}"
options="${opt[2]}"

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/segmentation-experiments

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

tensorboard --logdir="/scratch/anbil106/anbil106/SSL/segmentation-baselines" --host 0.0.0.0 --load_fast false &
python main.py --seed 42 --use-tensorboard --dataset "factin" \
    --backbone "resnet50" --backbone-weights ${weight} \
    --label-percentage ${subset} --opts ${options}

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"