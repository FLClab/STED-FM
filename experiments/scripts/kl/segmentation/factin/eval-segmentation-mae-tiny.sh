#!/usr/bin/env bash
#
#SBATCH --time=1:00:00
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
VENV_DIR=${HOME}/myenv
source $VENV_DIR/bin/activate
# source /home/frbea320/projects/def-flavielc/frbea320/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started evaluating segmentation model"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# echo "% from scratch"
# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# python eval.py --restore-from /home/koles2/scratch/ssl_project/segmentation_baselines/mae-lightning-tiny/factin/from-scratch/result.pt \
#  --dataset factin \
#  --backbone mae-lightning-tiny

# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# echo "% HPA - frozen"
# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# python eval.py \
#     --restore-from /home/koles2/scratch/ssl_project/segmentation_baselines/mae-lightning-tiny/factin/pretrained-frozen-MAE_TINY_HPA/result.pt \
#     --dataset factin \
#     --backbone mae-lightning-tiny
# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# echo "% HPA - pretrained"
# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# python eval.py \
#     --restore-from /home/koles2/scratch/ssl_project/segmentation_baselines/mae-lightning-tiny/factin/pretrained-MAE_TINY_HPA/result.pt \
#     --dataset factin \
#     --backbone mae-lightning-tiny 

# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# echo "% ImageNet - frozen"
# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# python eval.py \
#     --restore-from /home/koles2/scratch/ssl_project/segmentation_baselines/mae-lightning-tiny/factin/pretrained-frozen-MAE_TINY_IMAGENET1K_V1/result.pt \
#     --dataset factin \
#     --backbone mae-lightning-tiny 
# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# echo "% ImageNet - pretrained"
# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# python eval.py \
#     --restore-from /home/koles2/scratch/ssl_project/segmentation_baselines/mae-lightning-tiny/factin/pretrained-MAE_TINY_IMAGENET1K_V1/result.pt \
#     --dataset factin \
#     --backbone mae-lightning-tiny
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% STED - frozen"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
python eval.py \
    --restore-from /home/koles2/scratch/ssl_project/segmentation_baselines2/mae-lightning-tiny/factin/pretrained-frozen-MAE_TINY_STED/result.pt \
    --dataset factin \
    --backbone mae-lightning-tiny 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% STED - pretrained"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
python eval.py \
    --restore-from /home/koles2/scratch/ssl_project/segmentation_baselines2/mae-lightning-tiny/factin/pretrained-MAE_TINY_STED/result.pt \
    --dataset factin \
    --backbone mae-lightning-tiny 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"