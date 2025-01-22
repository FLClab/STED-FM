#!/bin/bash

cd evaluation

python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 42 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 43 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 44 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 45 --overwrite --from-scratch --opts "batch_size 64"  
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --blocks "all" --seed 46 --overwrite --from-scratch --opts "batch_size 64"