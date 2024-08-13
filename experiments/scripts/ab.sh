#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --mem=0
#SBATCH --nodes=2        
#SBATCH --gres=gpu:4   
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --array=0
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=koles2@ulaval.ca
#SBATCH --mail-type=ALL
#

export TORCH_NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.
# export MASTER_ADDR=$(hostname)
# export MASTER_PORT=42424

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
# VENV_DIR=${HOME}/venvs/ssl
VENV_DIR=${HOME}/myenv
source $VENV_DIR/bin/activate

# Moves to working directory
#cd ${HOME}/Documents/flc-dataset/experiments/simclr-experiments

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# cp "/project/def-flavielc/datasets/FLCDataset/dataset.tar" "${SLURM_TMPDIR}/dataset.tar"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

#tensorboard --logdir="/scratch/anbil106/anbil106/SSL/baselines" --host 0.0.0.0 --load_fast false &
tensorboard --logdir="./data/SSL/baselines/resnet18_hpa/lightning_logs" --host 0.0.0.0 --load_fast false &
#srun python main-lightning.py --seed 42 --use-tensorboard --dataset-path "${SLURM_TMPDIR}/dataset.tar" --backbone "resnet18" \
srun python main-lightning_v2.py --seed 42 --use-tensorboard --dataset-path "/home/koles2/scratch/hpa" --backbone "convnext-small" --restore-from "/home/koles2/projects/def-flavielc/koles2/flc-dataset/experiments/simclr-experiments/data/SSL/baselines/convnext-small_hpa/result.pt"
    # --restore-from "/scratch/anbil106/anbil106/SSL/baselines/resnet18/result.pt"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"