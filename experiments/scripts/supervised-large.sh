#!/usr/bin/env bash
#
#SBATCH --time=8:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=koles@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
#source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate
VENV_DIR=${HOME}/myenv
source $VENV_DIR/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WEIGHTS=(
    # "MAE_TINY_IMAGENET1K_V1"
    # "MAE_TINY_JUMP"
    # "MAE_TINY_STED"
    RESNET18_SSL_HPA
MODELS=(
    "mae-lightning-base"
    "mae-lightning-large"
)

model=${MODELS[${SLURM_ARRAY_TASK_ID}]}

# Moves to working directory
#cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation
cd ${HOME}/flc-dataset/experiments/evaluation
# Launch training 
cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started supervised training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

#python finetune.py --dataset synaptic-proteins --model resnet18 --weights $weight --blocks "all"
python finetune.py --dataset optim --model resnet18 --weights RESNET18_SSL_STED --blocks "all"
python supervised.py --dataset optim --model $model

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
