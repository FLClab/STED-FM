#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --mem=64Gb          
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

export TORCH_NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

module load python/3.10 scipy-stack/2023b
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp "/project/def-flavielc/datasets/FLCDataset/dataset.tar" "${SLURM_TMPDIR}/dataset.tar"

restore="./Datasets/FLCDataset/baselines/resnet18_STED/pl_current_model.pth"

savefolder="./Datasets/FLCDataset/baselines/resnet18_STED"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"



tensorboard --logdir="./Datasets/FLCDataset/baselines" --host 0.0.0.0 --load_fast false & 
srun python main_lightning.py --seed 42 --backbone resnet18 --dataset STED --use-tensorboard --save-folder $savefolder --dataset-path "${SLURM_TMPDIR}/dataset.tar"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
