#!/bin/bash

echo "=== Optim dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset optim
python knn.py --weights MAE_SMALL_JUMP --dataset optim
python knn.py --weights MAE_SMALL_HPA --dataset optim 
python knn.py --weights MAE_SMALL_SIM --dataset optim
python knn.py --weights MAE_SMALL_STED --dataset optim

echo "=== Neural Activity States dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset neural-activity-states
python knn.py --weights MAE_SMALL_JUMP --dataset neural-activity-states
python knn.py --weights MAE_SMALL_HPA --dataset neural-activity-states
python knn.py --weights MAE_SMALL_SIM --dataset neural-activity-states
python knn.py --weights MAE_SMALL_STED --dataset neural-activity-states

echo "=== Peroxisome dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset peroxisome
python knn.py --weights MAE_SMALL_JUMP --dataset peroxisome
python knn.py --weights MAE_SMALL_HPA --dataset peroxisome
python knn.py --weights MAE_SMALL_SIM --dataset peroxisome
python knn.py --weights MAE_SMALL_STED --dataset peroxisome

echo "=== Polymer Rings dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset polymer-rings
python knn.py --weights MAE_SMALL_JUMP --dataset polymer-rings
python knn.py --weights MAE_SMALL_HPA --dataset polymer-rings
python knn.py --weights MAE_SMALL_SIM --dataset polymer-rings
python knn.py --weights MAE_SMALL_STED --dataset polymer-rings

echo "=== DL-SIM dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset dl-sim
python knn.py --weights MAE_SMALL_JUMP --dataset dl-sim
python knn.py --weights MAE_SMALL_HPA --dataset dl-sim
python knn.py --weights MAE_SMALL_SIM --dataset dl-sim
python knn.py --weights MAE_SMALL_STED --dataset dl-sim
