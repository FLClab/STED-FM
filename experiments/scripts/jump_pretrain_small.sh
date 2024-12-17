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

cp "/home/frbea320/scratch/Datasets/JUMP_CP/jump.tar" "${SLURM_TMPDIR}/jump.tar"


restore="/home/frbea320/scratch/baselines/mae-small_JUMP/current_model.pth"

savefolder="/home/frbea320/scratch/baselines/mae-small_JUMP"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"


if test -e /home/frbea320/scratch/baselines/mae-small_JUMP/current_model.pth; then
    tensorboard --logdir="/home/frbea320/scratch/baselines/mae-small_JUMP" --host 0.0.0.0 --load_fast false & 
    srun python pretrain_lightning.py --seed 42 --model mae-lightning-small --dataset JUMP --use-tensorboard --save-folder $savefolder --dataset-path "${SLURM_TMPDIR}/jump.tar" --restore-from $restore 
else
    tensorboard --logdir="/home/frbea320/scratch/baselines/mae-small_JUMP" --host 0.0.0.0 --load_fast false & 
    srun python pretrain_lightning.py --seed 42 --model mae-lightning-small --dataset JUMP --use-tensorboard --save-folder $savefolder --dataset-path "${SLURM_TMPDIR}/jump.tar"
fi

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"