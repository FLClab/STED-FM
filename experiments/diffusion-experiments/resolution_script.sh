#!/bin/bash

python resolution_experiment.py --weights MAE_SMALL_STED
python resolution_experiment.py --weights MAE_SMALL_HPA
python resolution_experiment.py --weights MAE_SMALL_SIM
python resolution_experiment.py --weights MAE_SMALL_JUMP
python resolution_experiment.py --weights MAE_SMALL_IMAGENET1K_V1
