#!/bin/bash

python quality_experiment.py --weights MAE_SMALL_HPA
python quality_experiment.py --weights MAE_SMALL_STED
python quality_experiment.py --figure 
