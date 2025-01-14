#!/bin/bash

cd evaluation

python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "all" --seed 42 --overwrite
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "all" --seed 43 --overwrite
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "all" --seed 44 --overwrite
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "all" --seed 45 --overwrite
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --weights MAE_SMALL_IMAGENET1K_V1 --blocks "all" --seed 46 --overwrite
