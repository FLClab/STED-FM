#!/bin/bash

cd evaluation

python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 42 --overwrite --weights MAE_SMALL_STED
# python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# python finetune_v2.py --dataset optim --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID

# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 42 --overwrite --weights MAE_SMALL_HYBRID
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# # python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID

# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 42 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# python finetune_v2.py --dataset peroxisome --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID

# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 42 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID

# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 42 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 43 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 44 --overwrite --weights MAE_SMALL_HYBRID
# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 45 --overwrite --weights MAE_SMALL_HYBRID  
# python finetune_v2.py --dataset dl-sim --model mae-lightning-small --blocks "all" --seed 46 --overwrite --weights MAE_SMALL_HYBRID