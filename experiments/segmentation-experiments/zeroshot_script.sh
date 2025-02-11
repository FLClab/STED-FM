#!/bin/bash

python zeroshot.py --weights MAE_SMALL_IMAGENET1K_V1 
python zeroshot.py --weights MAE_SMALL_JUMP 
python zeroshot.py --weights MAE_SMALL_HPA
python zeroshot.py --weights MAE_SMALL_SIM 
python zeroshot.py --weights MAE_SMALL_STED 
