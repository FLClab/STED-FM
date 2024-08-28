#!/usr/bin/env bash
#
#SBATCH --time=12:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started finetuning"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

echo -e "==================== OPTIM ===================="

# python finetune_v2.py --dataset optim --model micranet --weights MICRANET_SSL_HPA --blocks 0
# python finetune_v2.py --dataset optim --model micranet --weights MICRANET_SSL_STED --blocks 0

# python finetune_v2.py --dataset optim --model resnet18 --weights RESNET18_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model resnet18 --weights RESNET18_SSL_HPA --blocks 0
python finetune_v2.py --dataset optim --model resnet18 --weights RESNET18_SSL_STED --blocks 0
python finetune_v2.py --dataset optim --model resnet18 --weights RESNET18_SSL_STED_ABLATION --blocks 0

# python finetune_v2.py --dataset optim --model resnet50 --weights RESNET50_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model resnet50 --weights RESNET50_SSL_HPA --blocks 0
# python finetune_v2.py --dataset optim --model resnet50 --weights RESNET50_SSL_STED --blocks 0

# python finetune_v2.py --dataset optim --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED --blocks 0

# python finetune_v2.py --dataset optim --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model convnext-small --weights CONVNEXT_SMALL_SSL_STED --blocks 0

# python finetune_v2.py --dataset optim --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model convnext-base --weights CONVNEXT_BASE_SSL_STED --blocks 0

# python finetune_v2.py --dataset optim --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model convnext-large --weights CONVNEXT_LARGE_SSL_STED --blocks 0

# python finetune_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_JUMP --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_HPA --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_STED --blocks 0

# python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_JUMP --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_HPA --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_STED --blocks 0

# python finetune_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_JUMP --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_HPA --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_STED --blocks 0

# python finetune_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_JUMP --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_HPA --blocks 0
# python finetune_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_STED --blocks 0

echo -e "\n\n==================== Neural Activity States ===================="

# python finetune_v2.py --dataset neural-activity-states --model micranet --weights MICRANET_SSL_HPA --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model micranet --weights MICRANET_SSL_STED --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model resnet18 --weights RESNET18_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model resnet18 --weights RESNET18_SSL_HPA --blocks 0
python finetune_v2.py --dataset neural-activity-states --model resnet18 --weights RESNET18_SSL_STED --blocks 0
python finetune_v2.py --dataset neural-activity-states --model resnet18 --weights RESNET18_SSL_STED_ABLATION --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model resnet50 --weights RESNET50_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model resnet50 --weights RESNET50_SSL_HPA --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model resnet50 --weights RESNET50_SSL_STED --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model convnext-small --weights CONVNEXT_SMALL_SSL_STED --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model convnext-base --weights CONVNEXT_BASE_SSL_STED --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model convnext-large --weights CONVNEXT_LARGE_SSL_STED --blocks 0


# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1  --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights MAE_TINY_JUMP --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights MAE_TINY_HPA --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-tiny --weights MAE_TINY_STED --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_JUMP --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_HPA --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_STED --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights MAE_BASE_JUMP --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights MAE_BASE_HPA --blocks 0
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-base --weights MAE_BASE_STED --blocks 0

# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1 --blocks 0 --opts batch_size 32
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-large --weights MAE_LARGE_JUMP --blocks 0 --opts batch_size 32
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-large --weights MAE_LARGE_HPA --blocks 0 --opts batch_size 32
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-large --weights MAE_LARGE_STED --blocks 0 --opts batch_size 32

echo -e "\n\n==================== Peroxisome ===================="

# python finetune_v2.py --dataset peroxisome --model micranet --weights MICRANET_SSL_HPA --blocks 0
# python finetune_v2.py --dataset peroxisome --model micranet --weights MICRANET_SSL_STED --blocks 0

# python finetune_v2.py --dataset peroxisome --model resnet18 --weights RESNET18_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model resnet18 --weights RESNET18_SSL_HPA --blocks 0
python finetune_v2.py --dataset peroxisome --model resnet18 --weights RESNET18_SSL_STED --blocks 0
python finetune_v2.py --dataset peroxisome --model resnet18 --weights RESNET18_SSL_STED_ABLATION --blocks 0

# python finetune_v2.py --dataset peroxisome --model resnet50 --weights RESNET50_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model resnet50 --weights RESNET50_SSL_HPA --blocks 0
# python finetune_v2.py --dataset peroxisome --model resnet50 --weights RESNET50_SSL_STED --blocks 0

# python finetune_v2.py --dataset peroxisome --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED --blocks 0

# python finetune_v2.py --dataset peroxisome --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model convnext-small --weights CONVNEXT_SMALL_SSL_STED --blocks 0

# python finetune_v2.py --dataset peroxisome --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model convnext-base --weights CONVNEXT_BASE_SSL_STED --blocks 0

# python finetune_v2.py --dataset peroxisome --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model convnext-large --weights CONVNEXT_LARGE_SSL_STED --blocks 0


# python finetune_v2.py --dataset peroxisome --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-tiny --weights MAE_TINY_JUMP --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-tiny --weights MAE_TINY_HPA --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-tiny --weights MAE_TINY_STED --blocks 0

# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights MAE_SMALL_JUMP --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights MAE_SMALL_HPA --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --weights MAE_SMALL_STED --blocks 0

# python finetune_v2.py --dataset peroxisome --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-base --weights MAE_BASE_JUMP --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-base --weights MAE_BASE_HPA --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-base --weights MAE_BASE_STED --blocks 0

# python finetune_v2.py --dataset peroxisome --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-large --weights MAE_LARGE_JUMP --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-large --weights MAE_LARGE_HPA --blocks 0
# python finetune_v2.py --dataset peroxisome --model mae-lightning-large --weights MAE_LARGE_STED --blocks 0

echo -e "\n\n==================== Polymer Rings ===================="

# python finetune_v2.py --dataset polymer-rings --model micranet --weights MICRANET_SSL_HPA --blocks 0
# python finetune_v2.py --dataset polymer-rings --model micranet --weights MICRANET_SSL_STED --blocks 0

# python finetune_v2.py --dataset polymer-rings --model resnet18 --weights RESNET18_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model resnet18 --weights RESNET18_SSL_HPA --blocks 0
python finetune_v2.py --dataset polymer-rings --model resnet18 --weights RESNET18_SSL_STED --blocks 0
python finetune_v2.py --dataset polymer-rings --model resnet18 --weights RESNET18_SSL_STED_ABLATION --blocks 0

# python finetune_v2.py --dataset polymer-rings --model resnet50 --weights RESNET50_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model resnet50 --weights RESNET50_SSL_HPA --blocks 0
# python finetune_v2.py --dataset polymer-rings --model resnet50 --weights RESNET50_SSL_STED --blocks 0

# python finetune_v2.py --dataset polymer-rings --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED --blocks 0

# python finetune_v2.py --dataset polymer-rings --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model convnext-small --weights CONVNEXT_SMALL_SSL_STED --blocks 0

# python finetune_v2.py --dataset polymer-rings --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model convnext-base --weights CONVNEXT_BASE_SSL_STED --blocks 0

# python finetune_v2.py --dataset polymer-rings --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model convnext-large --weights CONVNEXT_LARGE_SSL_STED --blocks 0


# python finetune_v2.py --dataset polymer-rings --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-tiny --weights MAE_TINY_JUMP --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-tiny --weights MAE_TINY_HPA --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-tiny --weights MAE_TINY_STED --blocks 0

# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --weights MAE_SMALL_JUMP --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --weights MAE_SMALL_HPA --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --weights MAE_SMALL_STED --blocks 0

# python finetune_v2.py --dataset polymer-rings --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-base --weights MAE_BASE_JUMP --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-base --weights MAE_BASE_HPA --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-base --weights MAE_BASE_STED --blocks 0

# python finetune_v2.py --dataset polymer-rings --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1 --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-large --weights MAE_LARGE_JUMP --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-large --weights MAE_LARGE_HPA --blocks 0
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-large --weights MAE_LARGE_STED --blocks 0

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
