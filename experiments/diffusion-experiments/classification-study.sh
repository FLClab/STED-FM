#!/bin/bash 


python classification_user_study.py --guidance class --ckpt-path "/home-local/Frederic/baselines/DiffusionModels/classifier-guidance" --dataset-path "/home-local/Frederic/Datasets/FLCDataset/dataset-250k.tar"
python classification_user_study.py --guidance latent --ckpt-path "/home-local/Frederic/baselines/DiffusionModels/latent-guidance" --dataset-path "/home-local/Frederic/Datasets/FLCDataset/dataset-250k.tar" --weights MAE_SMALL_IMAGENET1K_V1 
python classification_user_study.py --guidance latent --ckpt-path "/home-local/Frederic/baselines/DiffusionModels/latent-guidance" --dataset-path "/home-local/Frederic/Datasets/FLCDataset/dataset-250k.tar" --weights MAE_SMALL_JUMP
python classification_user_study.py --guidance latent --ckpt-path "/home-local/Frederic/baselines/DiffusionModels/latent-guidance" --dataset-path "/home-local/Frederic/Datasets/FLCDataset/dataset-250k.tar" --weights MAE_SMALL_HPA
python classification_user_study.py --guidance latent --ckpt-path "/home-local/Frederic/baselines/DiffusionModels/latent-guidance" --dataset-path "/home-local/Frederic/Datasets/FLCDataset/dataset-250k.tar" --weights MAE_SMALL_SIM
python classification_user_study.py --guidance latent --ckpt-path "/home-local/Frederic/baselines/DiffusionModels/latent-guidance" --dataset-path "/home-local/Frederic/Datasets/FLCDataset/dataset-250k.tar" --weights MAE_SMALL_STED
