#!/usr/bin/env bash
#
#SBATCH --time=00:30:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack
module load cuda cudnn
source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir $SLURM_TMPDIR/data

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

cp "/home/frbea320/scratch/Datasets/FLCDataset/TheresaProteins/theresa_proteins.hdf5" "${SLURM_TMPDIR}/data/theresa_proteins.hdf5"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done copy file"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "ResNet KNN classification"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python knn_classification.py --class-type protein --pretraining STED --datapath $SLURM_TMPDIR/data 


echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Done"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"