#!/bin/bash

echo "=== Optim dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset optim --pca
python knn.py --weights MAE_SMALL_JUMP --dataset optim --pca
python knn.py --weights MAE_SMALL_HPA --dataset optim --pca
python knn.py --weights MAE_SMALL_SIM --dataset optim --pca
python knn.py --weights MAE_SMALL_STED --dataset optim --pca

echo "=== Neural Activity States dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset neural-activity-states --pca
python knn.py --weights MAE_SMALL_JUMP --dataset neural-activity-states --pca
python knn.py --weights MAE_SMALL_HPA --dataset neural-activity-states --pca
python knn.py --weights MAE_SMALL_SIM --dataset neural-activity-states --pca
python knn.py --weights MAE_SMALL_STED --dataset neural-activity-states --pca

echo "=== Peroxisome dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset peroxisome --pca
python knn.py --weights MAE_SMALL_JUMP --dataset peroxisome --pca
python knn.py --weights MAE_SMALL_HPA --dataset peroxisome --pca
python knn.py --weights MAE_SMALL_SIM --dataset peroxisome --pca
python knn.py --weights MAE_SMALL_STED --dataset peroxisome --pca

echo "=== Polymer Rings dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset polymer-rings --pca
python knn.py --weights MAE_SMALL_JUMP --dataset polymer-rings --pca
python knn.py --weights MAE_SMALL_HPA --dataset polymer-rings --pca
python knn.py --weights MAE_SMALL_SIM --dataset polymer-rings --pca
python knn.py --weights MAE_SMALL_STED --dataset polymer-rings --pca

echo "=== DL-SIM dataset ==="
python knn.py --weights MAE_SMALL_IMAGENET1K_V1 --dataset dl-sim --pca
python knn.py --weights MAE_SMALL_JUMP --dataset dl-sim --pca
python knn.py --weights MAE_SMALL_HPA --dataset dl-sim --pca
python knn.py --weights MAE_SMALL_SIM --dataset dl-sim --pca
python knn.py --weights MAE_SMALL_STED --dataset dl-sim --pca
