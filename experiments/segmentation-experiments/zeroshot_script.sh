#!/bin/bash

python zeroshot.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset optim
python zeroshot.py --weights MAE_SMALL_JUMP --dataset optim
python zeroshot.py --weights MAE_SMALL_HPA --dataset optim
python zeroshot.py --weights MAE_SMALL_SIM --dataset optim
python zeroshot.py --weights MAE_SMALL_STED --dataset optim
