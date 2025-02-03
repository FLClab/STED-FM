#!/bin/bash

cd evaluation

python finetune_v2.py --dataset optim --model mae-lightning-small --seed 42 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset optim --model mae-lightning-small --seed 43 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset optim --model mae-lightning-small --seed 44 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset optim --model mae-lightning-small --seed 45 --overwrite --from-scratch --opts "batch_size 64"  
python finetune_v2.py --dataset optim --model mae-lightning-small --seed 46 --overwrite --from-scratch --opts "batch_size 64"

python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --seed 42 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --seed 43 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --seed 44 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --seed 45 --overwrite --from-scratch --opts "batch_size 64"  
python finetune_v2.py --dataset neural-activity-states --model mae-lightning-small --seed 46 --overwrite --from-scratch --opts "batch_size 64"

python finetune_v2.py --dataset peroxisome --model mae-lightning-small --seed 42 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset peroxisome --model mae-lightning-small --seed 43 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset peroxisome --model mae-lightning-small --seed 44 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset peroxisome --model mae-lightning-small --seed 45 --overwrite --from-scratch --opts "batch_size 64"  
python finetune_v2.py --dataset peroxisome --model mae-lightning-small --seed 46 --overwrite --from-scratch --opts "batch_size 64"

python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --seed 42 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --seed 43 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --seed 44 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --seed 45 --overwrite --from-scratch --opts "batch_size 64"  
python finetune_v2.py --dataset polymer-rings --model mae-lightning-small --seed 46 --overwrite --from-scratch --opts "batch_size 64"

python finetune_v2.py --dataset dl-sim --model mae-lightning-small --seed 42 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset dl-sim --model mae-lightning-small --seed 43 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset dl-sim --model mae-lightning-small --seed 44 --overwrite --from-scratch --opts "batch_size 64"
python finetune_v2.py --dataset dl-sim --model mae-lightning-small --seed 45 --overwrite --from-scratch --opts "batch_size 64"  
python finetune_v2.py --dataset dl-sim --model mae-lightning-small --seed 46 --overwrite --from-scratch --opts "batch_size 64"