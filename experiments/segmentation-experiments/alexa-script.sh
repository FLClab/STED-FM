#!/bin/bash

python main.py --dataset lioness --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --use-tensorboard --seed 46 --opts "batch_size 512"
