
MODELS=(
    "resnet18"
    "resnet50"
    "resnet101"
)
DATASETS=(
    "factin"
    "lioness"
    "footprocess"
)
MODES=(
    "from-scratch"
    "pretrained"
    "pretrained-frozen"
)
for model in "${MODELS[@]}";
do
    for dataset in "${DATASETS[@]}";
    do
        for mode in "${MODES[@]}";
        do
            python figure-small-dataset.py --model $model --dataset $dataset --samples 10 25 50 100 250 --mode $mode
        done
        python figure-scratch-pretrained.py --model $model --dataset $dataset
    done
done