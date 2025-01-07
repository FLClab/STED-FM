#!/bin/bash 

cd ./evaluation

python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_IMAGENET1K_V1" --blocks "all" --seed 42
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_IMAGENET1K_V1" --blocks "all" --seed 43
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_IMAGENET1K_V1" --blocks "all" --seed 44
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_IMAGENET1K_V1" --blocks "all" --seed 45
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_IMAGENET1K_V1" --blocks "all" --seed 46

python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_JUMP" --blocks "all" --seed 42
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_JUMP" --blocks "all" --seed 43
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_JUMP" --blocks "all" --seed 44
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_JUMP" --blocks "all" --seed 45
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_JUMP" --blocks "all" --seed 46

python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HPA" --blocks "all" --seed 42
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HPA" --blocks "all" --seed 43
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HPA" --blocks "all" --seed 44
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HPA" --blocks "all" --seed 45
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HPA" --blocks "all" --seed 46

python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_SIM" --blocks "all" --seed 42
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_SIM" --blocks "all" --seed 43
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_SIM" --blocks "all" --seed 44
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_SIM" --blocks "all" --seed 45
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_SIM" --blocks "all" --seed 46

python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_STED" --blocks "all" --seed 42
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_STED" --blocks "all" --seed 43
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_STED" --blocks "all" --seed 44
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_STED" --blocks "all" --seed 45
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_STED" --blocks "all" --seed 46

python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HYBRID" --blocks "all" --seed 42
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HYBRID" --blocks "all" --seed 43
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HYBRID" --blocks "all" --seed 44
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HYBRID" --blocks "all" --seed 45
python finetune_v2.py --dataset optim --model mae-lightning-small --weights "MAE_SMALL_HYBRID" --blocks "all" --seed 46
