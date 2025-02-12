#!/bin/bash

python patch_similarity.py --weights MAE_SMALL_IMAGENET1K_V1 --ckpt-path ~/../../home-local/Frederic/baselines/mae-small_ImageNet/optim/finetuned_None_44.pth
python patch_similarity.py --weights MAE_SMALL_JUMP --ckpt-path ~/../../home-local/Frederic/baselines/mae-small_JUMP/optim/finetuned_None_44.pth
python patch_similarity.py --weights MAE_SMALL_HPA --ckpt-path ~/../../home-local/Frederic/baselines/mae-small_HPA/optim/finetuned_None_44.pth
python patch_similarity.py --weights MAE_SMALL_SIM --ckpt-path ~/../../home-local/Frederic/baselines/mae-small_SIM/optim/finetuned_None_44.pth
python patch_similarity.py --weights MAE_SMALL_STED --ckpt-path ~/../../home-local/Frederic/baselines/mae-small_STED/optim/finetuned_None_44.pth
