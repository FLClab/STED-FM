#!/bin/bash
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --account=def-flavielc
#SBATCH --time=00:10:00
#SBATCH --mem=16Gb
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=frederic.beaupre.3@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8


module load python/3.8 scipy-stack
module load cuda cudnn

source /home/frbea320/projects/def-flavielc/frbea320/ad/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir $SLURM_TMPDIR/data

cp /home/frbea320/scratch/Datasets/FLCDataset/TheresaProteins/theresa_proteins.hdf5 $SLURM_TMPDIR/data


echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "KNN classification of synaptic proteins"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python knn_classification.py --datapath $SLURM_TMPDIR/data -ct protein


echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Done"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"