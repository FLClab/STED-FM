MODELS=(
    "resnet18"
    "resnet50"
    "resnet101"
)
DATASET=(
    "optim"
    "neural-activity-states"
    "peroxisome"
    "polymer-rings"
)
for dataset in ${DATASET[@]};
do
    python table-linear-probe-finetuned.py --dataset $dataset
done
for model in ${MODELS[@]};
do
    for dataset in ${DATASET[@]};
    do
        python figure-linear-probe-finetuned.py --model $model --dataset $dataset
        python figure-small-dataset.py --model $model --dataset $dataset --samples 10 25 50 100 250 500
    done
done