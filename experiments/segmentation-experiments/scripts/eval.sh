#!/usr/bin/env bash
#
#SBATCH --time=4:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

#### PARAMETERS

# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RESTOREFROM=(
    "/home/frbea320/projects/def-flavielc/segmentation-baselines/mae-lightning-small/factin/pretrained-frozen-MAE_SMALL_IMAGENET1K_V1/result.pt"
    "/home/frbea320/projects/def-flavielc/segmentation-baselines/mae-lightning-small/factin/pretrained-frozen-MAE_SMALL_HPA/result.pt"
    "/home/frbea320/projects/def-flavielc/segmentation-baselines/mae-lightning-small/factin/pretrained-frozen-MAE_SMALL_JUMP/result.pt"
    "/home/frbea320/projects/def-flavielc/segmentation-baselines/mae-lightning-small/factin/pretrained-frozen-MAE_SMALL_STED/result.pt"
)

restorefrom=${RESTOREFROM[${SLURM_ARRAY_TASK_ID}]}

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started evaluating segmentation model"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python eval.py --restore-from $restorefrom --dataset "factin" --opts batch_size 32

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
