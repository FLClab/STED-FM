#!/bin/bash

python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_IMAGENET1K_V1 --use-tensorboard --seed 42 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_IMAGENET1K_V1 --use-tensorboard --seed 43 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_IMAGENET1K_V1 --use-tensorboard --seed 44 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_IMAGENET1K_V1 --use-tensorboard --seed 45 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_IMAGENET1K_V1 --use-tensorboard --seed 46 --opts "batch_size 512"

python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_JUMP --use-tensorboard --seed 42 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_JUMP --use-tensorboard --seed 43 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_JUMP --use-tensorboard --seed 44 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_JUMP --use-tensorboard --seed 45 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_JUMP --use-tensorboard --seed 46 --opts "batch_size 512"

python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_HPA --use-tensorboard --seed 42 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_HPA --use-tensorboard --seed 43 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_HPA --use-tensorboard --seed 44 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_HPA --use-tensorboard --seed 45 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_HPA --use-tensorboard --seed 46 --opts "batch_size 512"

python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_SIM --use-tensorboard --seed 42 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_SIM --use-tensorboard --seed 43 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_SIM --use-tensorboard --seed 44 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_SIM --use-tensorboard --seed 45 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_SIM --use-tensorboard --seed 46 --opts "batch_size 512"

python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --use-tensorboard --seed 42 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --use-tensorboard --seed 43 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --use-tensorboard --seed 44 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --use-tensorboard --seed 45 --opts "batch_size 512"
python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --use-tensorboard --seed 46 --opts "batch_size 512"
