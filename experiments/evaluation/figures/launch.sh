MODELS=(
    "mae-small"
)
DATASET=(
    "optim"
    "neural-activity-states"
    "peroxisome"
    "polymer-rings"
    "dl-sim"
    "bbbc026"
    "bbbc052"
    "bbbc053"
    "hpa-classification"
)
for model in ${MODELS[@]};
do
    for dataset in ${DATASET[@]};
    do  
        metric="acc"
        if [ "$dataset" == "hpa-classification" ]; then
            metric="f1"
        fi
        python figure-small-dataset.py --model $model --dataset $dataset --samples 10 25 50 100 --mode linear-probe --metric $metric
        python figure-small-dataset.py --model $model --dataset $dataset --samples 10 25 50 100 --mode finetuned --metric $metric
    done
done