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
#SBATCH --array=0-299

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Training options
BACKBONEWEIGHTS=(
    "MAE_SMALL_IMAGENET1K_V1"
    "MAE_SMALL_JUMP"
    "MAE_SMALL_HPA"
    "MAE_SMALL_SIM"
    "MAE_SMALL_STED"    
)

DATASETS=(
    "factin"
    "footprocess"
    "lioness"
)

SEEDS=(
    42
    43
    44
    45
    46
)

SUBSETS=(
    "0.01"
    "0.10"
    "0.25"
    "0.50"
)

opts=()
for dataset in "${DATASETS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        for weight in "${BACKBONEWEIGHTS[@]}"
        do
            for subset in "${SUBSETS[@]}"
            do
                opts+=("$dataset;$seed;$weight;$subset")
            done
        done
    done
done

IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
dataset="${opt[0]}"
seed="${opt[1]}"
weight="${opt[2]}"
subset="${opt[3]}"
cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/segmentation-experiments


# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "% Seed: ${seed}"
echo "% Weight: ${weight}"
echo "% Dataset: ${dataset}"
echo "% Subset: ${subset}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python main.py --seed $seed --dataset $dataset --backbone "mae-lightning-small" --backbone-weights $weight --label-percentage $subset --opts "freeze_backbone true batch_size 128"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"