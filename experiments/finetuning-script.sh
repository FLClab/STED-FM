#!/bin/bash

cd evaluation

python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 42 --overwrite --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 43 --overwrite --opts "batch_size 64"      
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 44 --overwrite --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 45 --overwrite --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "0" --seed 46 --overwrite --opts "batch_size 64"
