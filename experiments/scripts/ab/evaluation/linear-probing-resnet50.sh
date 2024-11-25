#!/usr/bin/env bash
#
#SBATCH --time=08:00:00
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
echo "% Started linear probing"
echo "% dataset: ${dataset}"
echo "% seed: ${seed}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_IMAGENET1K_V1 --seed $seed
python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SSL_HPA --seed $seed
python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SSL_JUMP --seed $seed
python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SSL_STED --seed $seed

# echo -e "==================== OPTIM ===================="

# # python finetune_v2.py --dataset optim --model micranet --weights MICRANET_SSL_HPA
# # python finetune_v2.py --dataset optim --model micranet --weights MICRANET_SSL_STED

# # python finetune_v2.py --dataset optim --model resnet18 --weights RESNET18_IMAGENET1K_V1 --seed $seed
# # python finetune_v2.py --dataset optim --model resnet18 --weights RESNET18_SSL_HPA --seed $seed
# # python finetune_v2.py --dataset optim --model resnet18 --weights RESNET18_SSL_JUMP --seed $seed
# # python finetune_v2.py --dataset optim --model resnet18 --weights RESNET18_SSL_STED --seed $seed

# python finetune_v2.py --dataset optim --model resnet50 --weights RESNET50_IMAGENET1K_V1 --seed $seed
# python finetune_v2.py --dataset optim --model resnet50 --weights RESNET50_SSL_JUMP --seed $seed
# python finetune_v2.py --dataset optim --model resnet50 --weights RESNET50_SSL_HPA --seed $seed
# python finetune_v2.py --dataset optim --model resnet50 --weights RESNET50_SSL_STED --seed $seed

# # python finetune_v2.py --dataset optim --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1
# # python finetune_v2.py --dataset optim --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED

# # python finetune_v2.py --dataset optim --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset optim --model convnext-small --weights CONVNEXT_SMALL_SSL_STED

# # python finetune_v2.py --dataset optim --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset optim --model convnext-base --weights CONVNEXT_BASE_SSL_STED

# # python finetune_v2.py --dataset optim --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset optim --model convnext-large --weights CONVNEXT_LARGE_SSL_STED

# # python finetune_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1
# # python finetune_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_JUMP
# # python finetune_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_HPA
# # python finetune_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_STED

# # python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_JUMP
# # python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_HPA
# # python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_STED

# # python finetune_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_JUMP
# # python finetune_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_HPA
# # python finetune_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_STED

# # python finetune_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_JUMP
# # python finetune_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_HPA
# # python finetune_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_STED


# echo -e "\n\n==================== Synaptic Proteins ===================="

# # python finetune_v2.py --dataset synaptic-proteins --model micranet --weights MICRANET_SSL_HPA
# # python finetune_v2.py --dataset synaptic-proteins --model micranet --weights MICRANET_SSL_STED

# # python finetune_v2.py --dataset synaptic-proteins --model resnet18 --weights RESNET18_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model resnet18 --weights RESNET18_SSL_HPA
# # python finetune_v2.py --dataset synaptic-proteins --model resnet18 --weights RESNET18_SSL_STED


# # python finetune_v2.py --dataset synaptic-proteins --model resnet50 --weights RESNET50_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model resnet50 --weights RESNET50_SSL_HPA
# # python finetune_v2.py --dataset synaptic-proteins --model resnet50 --weights RESNET50_SSL_STED

# # python finetune_v2.py --dataset synaptic-proteins --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED

# # python finetune_v2.py --dataset synaptic-proteins --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model convnext-small --weights CONVNEXT_SMALL_SSL_STED

# # python finetune_v2.py --dataset synaptic-proteins --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model convnext-base --weights CONVNEXT_BASE_SSL_STED

# # python finetune_v2.py --dataset synaptic-proteins --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model convnext-large --weights CONVNEXT_LARGE_SSL_STED


# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1 
# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-tiny --weights MAE_TINY_JUMP
# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-tiny --weights MAE_TINY_STED

# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-small --weights MAE_SMALL_JUMP
# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-small --weights MAE_SMALL_STED

# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-base --weights MAE_BASE_JUMP
# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-base --weights MAE_BASE_STED

# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-large --weights MAE_LARGE_JUMP
# # python finetune_v2.py --dataset synaptic-proteins --model mae-lightning-large --weights MAE_LARGE_STED

# echo -e "\n\n==================== Neural Activity States ===================="

# # python finetune_v2.py --dataset neural-activity-states --model micranet --weights MICRANET_SSL_HPA
# # python finetune_v2.py --dataset neural-activity-states --model micranet --weights MICRANET_SSL_STED

# # python finetune_v2.py --dataset neural-activity-states --model resnet18 --weights RESNET18_IMAGENET1K_V1 --seed $seed
# # python finetune_v2.py --dataset neural-activity-states --model resnet18 --weights RESNET18_SSL_HPA --seed $seed
# # python finetune_v2.py --dataset neural-activity-states --model resnet18 --weights RESNET18_SSL_JUMP --seed $seed
# # python finetune_v2.py --dataset neural-activity-states --model resnet18 --weights RESNET18_SSL_STED --seed $seed

# python finetune_v2.py --dataset neural-activity-states --model resnet50 --weights RESNET50_IMAGENET1K_V1 --seed $seed
# python finetune_v2.py --dataset neural-activity-states --model resnet50 --weights RESNET50_SSL_HPA --seed $seed
# python finetune_v2.py --dataset neural-activity-states --model resnet50 --weights RESNET50_SSL_JUMP --seed $seed
# python finetune_v2.py --dataset neural-activity-states --model resnet50 --weights RESNET50_SSL_STED --seed $seed

# # python finetune_v2.py --dataset neural-activity-states --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1
# # python finetune_v2.py --dataset neural-activity-states --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED

# # python finetune_v2.py --dataset neural-activity-states --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset neural-activity-states --model convnext-small --weights CONVNEXT_SMALL_SSL_STED

# # python finetune_v2.py --dataset neural-activity-states --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset neural-activity-states --model convnext-base --weights CONVNEXT_BASE_SSL_STED

# # python finetune_v2.py --dataset neural-activity-states --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset neural-activity-states --model convnext-large --weights CONVNEXT_LARGE_SSL_STED


# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1 
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights MAE_TINY_JUMP
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights MAE_TINY_HPA
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights MAE_TINY_STED

# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_JUMP
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_HPA
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_STED

# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights MAE_BASE_JUMP
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights MAE_BASE_HPA
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights MAE_BASE_STED

# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-large --weights MAE_LARGE_JUMP
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-large --weights MAE_LARGE_HPA
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-large --weights MAE_LARGE_STED

# echo -e "\n\n==================== Peroxisome ===================="

# # python finetune_v2.py --dataset peroxisome --model micranet --weights MICRANET_SSL_HPA
# # python finetune_v2.py --dataset peroxisome --model micranet --weights MICRANET_SSL_STED

# # python finetune_v2.py --dataset peroxisome --model resnet18 --weights RESNET18_IMAGENET1K_V1 --seed $seed
# # python finetune_v2.py --dataset peroxisome --model resnet18 --weights RESNET18_SSL_HPA --seed $seed
# # python finetune_v2.py --dataset peroxisome --model resnet18 --weights RESNET18_SSL_JUMP --seed $seed
# # python finetune_v2.py --dataset peroxisome --model resnet18 --weights RESNET18_SSL_STED --seed $seed

# python finetune_v2.py --dataset peroxisome --model resnet50 --weights RESNET50_IMAGENET1K_V1 --seed $seed
# python finetune_v2.py --dataset peroxisome --model resnet50 --weights RESNET50_SSL_HPA --seed $seed
# python finetune_v2.py --dataset peroxisome --model resnet50 --weights RESNET50_SSL_JUMP --seed $seed
# python finetune_v2.py --dataset peroxisome --model resnet50 --weights RESNET50_SSL_STED --seed $seed

# # python finetune_v2.py --dataset peroxisome --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1
# # python finetune_v2.py --dataset peroxisome --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED

# # python finetune_v2.py --dataset peroxisome --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset peroxisome --model convnext-small --weights CONVNEXT_SMALL_SSL_STED

# # python finetune_v2.py --dataset peroxisome --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset peroxisome --model convnext-base --weights CONVNEXT_BASE_SSL_STED

# # python finetune_v2.py --dataset peroxisome --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset peroxisome --model convnext-large --weights CONVNEXT_LARGE_SSL_STED


# # python finetune_v2.py --dataset peroxisome --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1 
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-tiny --weights MAE_TINY_JUMP
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-tiny --weights MAE_TINY_HPA
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-tiny --weights MAE_TINY_STED

# # python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights MAE_SMALL_JUMP
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights MAE_SMALL_HPA
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights MAE_SMALL_STED

# # python finetune_v2.py --dataset peroxisome --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-base --weights MAE_BASE_JUMP
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-base --weights MAE_BASE_HPA
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-base --weights MAE_BASE_STED

# # python finetune_v2.py --dataset peroxisome --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-large --weights MAE_LARGE_JUMP
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-large --weights MAE_LARGE_HPA
# # python finetune_v2.py --dataset peroxisome --model mae-lightning-large --weights MAE_LARGE_STED

# echo -e "\n\n==================== Polymer Rings ===================="

# # python finetune_v2.py --dataset polymer-rings --model micranet --weights MICRANET_SSL_HPA
# # python finetune_v2.py --dataset polymer-rings --model micranet --weights MICRANET_SSL_STED

# # python finetune_v2.py --dataset polymer-rings --model resnet18 --weights RESNET18_IMAGENET1K_V1 --seed $seed
# # python finetune_v2.py --dataset polymer-rings --model resnet18 --weights RESNET18_SSL_HPA --seed $seed
# # python finetune_v2.py --dataset polymer-rings --model resnet18 --weights RESNET18_SSL_JUMP --seed $seed
# # python finetune_v2.py --dataset polymer-rings --model resnet18 --weights RESNET18_SSL_STED --seed $seed

# python finetune_v2.py --dataset polymer-rings --model resnet50 --weights RESNET50_IMAGENET1K_V1 --seed $seed
# python finetune_v2.py --dataset polymer-rings --model resnet50 --weights RESNET50_SSL_HPA --seed $seed
# python finetune_v2.py --dataset polymer-rings --model resnet50 --weights RESNET50_SSL_JUMP --seed $seed
# python finetune_v2.py --dataset polymer-rings --model resnet50 --weights RESNET50_SSL_STED --seed $seed

# # python finetune_v2.py --dataset polymer-rings --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1
# # python finetune_v2.py --dataset polymer-rings --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED

# # python finetune_v2.py --dataset polymer-rings --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset polymer-rings --model convnext-small --weights CONVNEXT_SMALL_SSL_STED

# # python finetune_v2.py --dataset polymer-rings --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset polymer-rings --model convnext-base --weights CONVNEXT_BASE_SSL_STED

# # python finetune_v2.py --dataset polymer-rings --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset polymer-rings --model convnext-large --weights CONVNEXT_LARGE_SSL_STED


# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1 
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-tiny --weights MAE_TINY_JUMP
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-tiny --weights MAE_TINY_HPA
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-tiny --weights MAE_TINY_STED

# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --weights MAE_SMALL_JUMP
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --weights MAE_SMALL_HPA
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --weights MAE_SMALL_STED

# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-base --weights MAE_BASE_JUMP
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-base --weights MAE_BASE_HPA
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-base --weights MAE_BASE_STED

# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-large --weights MAE_LARGE_JUMP
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-large --weights MAE_LARGE_HPA
# # python finetune_v2.py --dataset polymer-rings --model mae-lightning-large --weights MAE_LARGE_STED

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
