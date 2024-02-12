#!/bin/bash
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --account=def-flavielc
#SBATCH --time=2:00:00
#SBATCH --mem=32Gb
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=frederic.beaupre.3@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8


module load python/3.8 scipy-stack
module load cuda cudnn

source /home/frbea320/projects/def-flavielc/frbea320/ad/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir $SLURM_TMPDIR/data

# We copy our large datasets to the compute node's local storage
cp /home/frbea320/scratch/Datasets/FLCDataset/dataset.tar $SLURM_TMPDIR/data

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Variational Autoencoder training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python train_mae.py --datapath $SLURM_TMPDIR/data

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Done"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
