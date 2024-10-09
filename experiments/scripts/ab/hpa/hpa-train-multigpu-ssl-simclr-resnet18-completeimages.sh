#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --gres=gpu:p100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4%1
#SBATCH --output=/home/anbil106/logs/%x-%A_%a.out
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL
#

export TORCH_NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/simclr-experiments

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp "/project/def-flavielc/datasets/train.zip" "${SLURM_TMPDIR}/dataset.zip"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# Launch training
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

tensorboard --logdir="/scratch/anbil106/projects/SSL/baselines/dataset-fullimages-1Msteps-multigpu/resnet18_HPA" --host 0.0.0.0 --load_fast false &

CKPT="/home/anbil106/scratch/projects/SSL/baselines/dataset-fullimages-1Msteps-multigpu/resnet18_HPA/result.pt"
if [ -f $CKPT ]; then
    echo "% Training from previous checkpoint: ${CKPT}"

    srun python main-lightning.py --seed 42 --use-tensorboard --dataset-path "${SLURM_TMPDIR}/dataset.zip" --backbone "resnet18" \
                                  --dataset "HPA" \
                                  --save-folder "./data/SSL/baselines/dataset-fullimages-1Msteps-multigpu" \
                                  --opts "batch_size 128" \
                                  --restore-from "${CKPT}"
else
    echo "% Training from scratch"

    srun python main-lightning.py --seed 42 --use-tensorboard --dataset-path "${SLURM_TMPDIR}/dataset.zip" --backbone "resnet18" \
                                  --dataset "HPA" \
                                  --save-folder "./data/SSL/baselines/dataset-fullimages-1Msteps-multigpu" \
                                  --opts "batch_size 128"
fi

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
