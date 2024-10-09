#!/usr/bin/env bash
#
#SBATCH --time=8:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=anbil106@ulaval.ca
#SBATCH --mail-type=ALL

#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.10 scipy-stack/2023b
VENV_DIR=${HOME}/venvs/ssl
source $VENV_DIR/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Moves to working directory
cd ${HOME}/Documents/flc-dataset/experiments/evaluation

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started Evaluation"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

echo -e "==================== OPTIM ===================="

# python eval_v2.py --dataset optim --model micranet --pretraining STED --probe linear-probe
# python eval_v2.py --dataset optim --model micranet --pretraining STED --probe finetuned

# python eval_v2.py --dataset optim --model resnet18 --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset optim --model resnet18 --pretraining ImageNet --probe finetuned
# python eval_v2.py --dataset optim --model resnet18 --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset optim --model resnet18 --pretraining HPA --probe finetuned
# python eval_v2.py --dataset optim --model resnet18 --pretraining STED_nonoise --probe linear-probe
# python eval_v2.py --dataset optim --model resnet18 --pretraining STED_nonoise --probe finetuned

# python eval_v2.py --dataset optim --model resnet50 --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset optim --model resnet50 --pretraining ImageNet --probe finetuned
# python eval_v2.py --dataset optim --model resnet50 --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset optim --model resnet50 --pretraining HPA --probe finetuned
# python eval_v2.py --dataset optim --model resnet50 --pretraining STED --probe linear-probe
# python eval_v2.py --dataset optim --model resnet50 --pretraining STED --probe finetuned

python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining HPA --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining JUMP --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining JUMP --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining STED --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-tiny --pretraining STED --probe finetuned

python eval_v2.py --dataset optim --model mae-lightning-small --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-small --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-small --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-small --pretraining HPA --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-small --pretraining JUMP --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-small --pretraining JUMP --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-small --pretraining STED --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-small --pretraining STED --probe finetuned

python eval_v2.py --dataset optim --model mae-lightning-base --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-base --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-base --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-base --pretraining HPA --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-base --pretraining JUMP --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-base --pretraining JUMP --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-base --pretraining STED --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-base --pretraining STED --probe finetuned

python eval_v2.py --dataset optim --model mae-lightning-large --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-large --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-large --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-large --pretraining HPA --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-large --pretraining JUMP --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-large --pretraining JUMP --probe finetuned
python eval_v2.py --dataset optim --model mae-lightning-large --pretraining STED --probe linear-probe
# python eval_v2.py --dataset optim --model mae-lightning-large --pretraining STED --probe finetuned

echo -e "\n\n==================== Neural Activity States ===================="

# python eval_v2.py --dataset neural-activity-states --model micranet --pretraining STED --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model micranet --pretraining STED --probe finetuned

python eval_v2.py --dataset neural-activity-states --model resnet18 --pretraining ImageNet --probe linear-probe
python eval_v2.py --dataset neural-activity-states --model resnet18 --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset neural-activity-states --model resnet18 --pretraining HPA --probe linear-probe
python eval_v2.py --dataset neural-activity-states --model resnet18 --pretraining HPA --probe finetuned
python eval_v2.py --dataset neural-activity-states --model resnet18 --pretraining STED_nonoise --probe linear-probe
python eval_v2.py --dataset neural-activity-states --model resnet18 --pretraining STED_nonoise --probe finetuned

# python eval_v2.py --dataset neural-activity-states --model resnet50 --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model resnet50 --pretraining ImageNet --probe finetuned
# python eval_v2.py --dataset neural-activity-states --model resnet50 --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model resnet50 --pretraining HPA --probe finetuned
# python eval_v2.py --dataset neural-activity-states --model resnet50 --pretraining STED --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model resnet50 --pretraining STED --probe finetuned

python eval_v2.py --dataset neural-activity-states --model mae-lightning-tiny --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-tiny --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-tiny --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-tiny --pretraining HPA --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-tiny --pretraining JUMP --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-tiny --pretraining JUMP --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-tiny --pretraining STED --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-tiny --pretraining STED --probe finetuned

python eval_v2.py --dataset neural-activity-states --model mae-lightning-small --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-small --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-small --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-small --pretraining HPA --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-small --pretraining JUMP --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-small --pretraining JUMP --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-small --pretraining STED --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-small --pretraining STED --probe finetuned

python eval_v2.py --dataset neural-activity-states --model mae-lightning-base --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-base --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-base --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-base --pretraining HPA --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-base --pretraining JUMP --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-base --pretraining JUMP --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-base --pretraining STED --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-base --pretraining STED --probe finetuned

python eval_v2.py --dataset neural-activity-states --model mae-lightning-large --pretraining ImageNet --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-large --pretraining ImageNet --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-large --pretraining HPA --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-large --pretraining HPA --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-large --pretraining JUMP --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-large --pretraining JUMP --probe finetuned
python eval_v2.py --dataset neural-activity-states --model mae-lightning-large --pretraining STED --probe linear-probe
# python eval_v2.py --dataset neural-activity-states --model mae-lightning-large --pretraining STED --probe finetuned

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
