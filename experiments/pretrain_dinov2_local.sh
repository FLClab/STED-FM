#!/bin/bash
# ── Activate virtual environment ───────────────────────────────────────────
source /home/mathis/Desktop/sted-env/bin/activate

# ── Parameters ─────────────────────────────────────────────────────────────
MODEL="dinov2-lightning-small"
DATASET="STED"
DATASET_PATH="/home-local/mathis/STED-FM/STED-FM-dataset-crops.tar"
SAVE_FOLDER="/home-local/mathis/STED-FM/baselines/dinov2-small_STED"
SEED=42

# ── Run ────────────────────────────────────────────────────────────────────
python experiments/pretrain_dinov2_lightning.py \
    --seed $SEED \
    --model $MODEL \
    --dataset $DATASET \
    --dataset-path $DATASET_PATH \
    --save-folder $SAVE_FOLDER \
    --dry-run
