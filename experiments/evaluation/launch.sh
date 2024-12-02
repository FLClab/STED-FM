
DATASETS=(
    "optim"
    "neural-activity-states"
    "polymer-rings"
    "peroxisome"
)
SEEDS=(
    42
    43
    44
    45
    46
)

for dataset in "${DATASETS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_IMAGENET1K_V1 --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_SSL_HPA --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_SSL_JUMP --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_SSL_STED --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_IMAGENET1K_V1 --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SSL_HPA --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SSL_JUMP --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SSL_STED --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_IMAGENET1K_V1 --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SSL_HPA --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SSL_JUMP --seed $seed --blocks "0" --overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SSL_STED --seed $seed --blocks "0" --overwrite
    done
done