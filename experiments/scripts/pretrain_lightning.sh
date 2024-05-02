#!/usr/bin/env bash
#

#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --mem=0
#SBATCH --nodes 1             
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4   
#SBATCH --cpus-per-task=10
#SBATCH --array=1-14%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp "./Datasets/FLCDataset/dataset.tar" "${SLURM_TMPDIR}/dataset.tar"

restore="./Datasets/FLCDataset/baselines/mae-small_STED/pl_current_model.pth"

savefolder="./Datasets/FLCDataset/baselines/mae-small_STED"

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"


if test -e ./Datasets/FLCDataset/baselines/mae-small_STED/pl_current_model.pth; then
    python pretrain_lightning.py --seed 42 --use-tensorboard --save-folder $savefolder --dataset-path "${SLURM_TMPDIR}/dataset.tar" --restore-from $restore 
else
    python pretrain_lightning.py --seed 42 --use-tensorboard --save-folder $savefolder --dataset-path "${SLURM_TMPDIR}/dataset.tar"
fi

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"