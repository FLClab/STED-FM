#!/bin/bash

savefolder="/home-local/Frederic/baselines/mae-small_64-p8"

python pretrain_lightning.py --seed 42 --model mae-lightning-64-p8 --dataset protein-diffusion --save-folder $savefolder --dataset-path "/home-local/Frederic/Datasets/LargeProteinModels"


