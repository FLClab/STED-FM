#!/bin/bash

cd evaluation

python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_STED --blocks "0" --seed 42 --opts "batch_size 64" --flops
python finetune_v2.py --dataset optim --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 43 --opts "batch_size 64" --flops      
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 44 --overwrite --opts "batch_size 64"
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 45 --overwrite --opts "batch_size 64"
# python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 46 --overwrite --opts "batch_size 64"
