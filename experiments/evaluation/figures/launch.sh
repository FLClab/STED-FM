MODELS=(
    "resnet18"
    "resnet50"
)
DATASET=(
    "optim"
    "neural-activity-states"
    "peroxisome"
    "polymer-rings"
)
for model in ${MODELS[@]};
do
    for dataset in ${DATASET[@]};
    do
        python figure-linear-probe-finetuned.py --model $model --dataset $dataset
        python figure-small-dataset.py --model $model --dataset $dataset --samples 10 25 50 100 250 500
    done
done