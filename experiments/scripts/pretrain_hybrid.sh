#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --mem=0
#SBATCH --nodes=1             
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4   
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

export TORCH_NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

module load python/3.10 scipy-stack/2023b
module load cuda cudnn
source ~/phd/bin/activate

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp "/project/def-flavielc/datasets/sim-dataset-crops.tar" "${SLURM_TMPDIR}/sim-dataset-crops.tar"
cp "/project/def-flavielc/datasets/train.zip" "${SLURM_TMPDIR}/train.zip"
cp "/project/def-flavielc/datasets/FLCDataset/dataset.tar" "${SLURM_TMPDIR}/dataset.tar"

restore="/home/frbea320/scratch/baselines/mae-small_Hybrid/pl_current_model.pth"

savefolder="/home/frbea320/scratch/baselines/mae-small_Hybrid"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"


if test -e ./Datasets/FLCDataset/baselines/mae-small_Hybrid/pl_current_model.pth; then
    tensorboard --logdir="./Datasets/FLCDataset/baselines" --host 0.0.0.0 --load_fast false & 
    srun python pretrain_lightning.py --seed 42 --model mae-lightning-small --dataset Hybrid --use-tensorboard --save-folder $savefolder --restore-from $restore --hpa-path "${SLURM_TMPDIR}/train.zip" --sim-path "${SLURM_TMPDIR}/sim-dataset-crops.tar" --sted-path "${SLURM_TMPDIR}/dataset.tar"
else
    tensorboard --logdir="./Datasets/FLCDataset/baselines" --host 0.0.0.0 --load_fast false & 
    srun python pretrain_lightning.py --seed 42 --model mae-lightning-small --dataset Hybrid --use-tensorboard --save-folder $savefolder --hpa-path "${SLURM_TMPDIR}/train.zip" --sim-path "${SLURM_TMPDIR}/sim-dataset-crops.tar" --sted-path "${SLURM_TMPDIR}/dataset.tar"
fi

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
