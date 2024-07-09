#!/usr/bin/env bash
#
#SBATCH --time=01:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started KNN classification"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

echo "==================== OPTIM ===================="
# python knn_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1
# python knn_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_JUMP
# python knn_v2.py --dataset optim --model mae-lightning-tiny --weights MAE_TINY_STED

# python knn_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1
# python knn_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_JUMP
# python knn_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_STED

# python knn_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1
# python knn_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_JUMP
# python knn_v2.py --dataset optim --model mae-lightning-base --weights MAE_BASE_STED

# python knn_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1
# python knn_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_JUMP
# python knn_v2.py --dataset optim --model mae-lightning-large --weights MAE_LARGE_STED


echo "==================== Synaptic Proteins ===================="

# python knn_v2.py --dataset synaptic-proteins --model micranet --weights MICRANET_SSL_STED

# python knn_v2.py --dataset synaptic-proteins --model resnet18 --weights RESNET18_IMAGENET1K_V1
# python knn_v2.py --dataset synaptic-proteins --model resnet18 --weights RESNET18_SSL_STED


# python knn_v2.py --dataset synaptic-proteins --model resnet50 --weights RESNET50_IMAGENET1K_V1
# python knn_v2.py --dataset synaptic-proteins --model resnet50 --weights RESNET50_SSL_STED

# python knn_v2.py --dataset synaptic-proteins --model convnext-tiny --weights CONVNEXT_TINY_IMAGENET1K_V1
# python knn_v2.py --dataset synaptic-proteins --model convnext-tiny --weights CONVNEXT_TINY_SSL_STED

# python knn_v2.py --dataset synaptic-proteins --model convnext-small --weights CONVNEXT_SMALL_IMAGENET1K_V1
# python knn_v2.py --dataset synaptic-proteins --model convnext-small --weights CONVNEXT_SMALL_SSL_STED

# python knn_v2.py --dataset synaptic-proteins --model convnext-base --weights CONVNEXT_BASE_IMAGENET1K_V1
# python knn_v2.py --dataset synaptic-proteins --model convnext-base --weights CONVNEXT_BASE_SSL_STED

# python knn_v2.py --dataset synaptic-proteins --model convnext-large --weights CONVNEXT_LARGE_IMAGENET1K_V1
# python knn_v2.py --dataset synaptic-proteins --model convnext-large --weights CONVNEXT_LARGE_SSL_STED


python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-tiny --weights MAE_TINY_IMAGENET1K_V1 
python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-tiny --weights MAE_TINY_JUMP
python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-tiny --weights MAE_TINY_STED

python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1
python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-small --weights MAE_SMALL_JUMP
python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-small --weights MAE_SMALL_STED

python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-base --weights MAE_BASE_IMAGENET1K_V1
python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-base --weights MAE_BASE_JUMP
python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-base --weights MAE_BASE_STED

python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-large --weights MAE_LARGE_IMAGENET1K_V1
python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-large --weights MAE_LARGE_JUMP
python knn_v2.py --dataset synaptic-proteins --label conditions --model mae-lightning-large --weights MAE_LARGE_STED

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
