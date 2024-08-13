#!/usr/bin/env bash
#
#SBATCH --time=0:30:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=koles2@ulaval.ca
#SBATCH --mail-type=ALL

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
#source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate
VENV_DIR=${HOME}/myenv
source $VENV_DIR/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PRETRAIN=(
    "ImageNet"
    "JUMP"
    "STED"
)


pretraining=${PRETRAIN[${SLURM_ARRAY_TASK_ID}]}

#cd ${HOME}/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation
cd ${HOME}/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started evaluation on the test set"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining $pretraining --probe linear-probe
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
