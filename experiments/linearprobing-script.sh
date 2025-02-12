#!/bin/bash

cd evaluation

python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 42 --weights MAE_SMALL_STED --flops
python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 43 --weights MAE_SMALL_SIM --flops
python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 44 --weights MAE_SMALL_HPA --flops
python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 45 --weights MAE_SMALL_JUMP --flops
python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 46 --weights MAE_SMALL_IMAGENET1K_V1 --flops

#python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 42 --weights MAE_SMALL_STED --flops
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID

#python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 42 --weights MAE_SMALL_STED --flops
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID

#python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 42 --weights MAE_SMALL_STED --flops
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID

# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 42 --weights MAE_SMALL_STED --flops
# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID
