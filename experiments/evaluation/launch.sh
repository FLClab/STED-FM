
DATASETS=(
    # "optim"
    # "neural-activity-states"
    # "polymer-rings"
    # "peroxisome"
    "dl-sim"
)
SEEDS=(
    42
    43
    44
    45
    46
)

for seed in "${SEEDS[@]}"
do
    for dataset in "${DATASETS[@]}"
    do
        python finetune_v2.py --dataset $dataset --model resnet18 --seed $seed --from-scratch #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_IMAGENET1K_V1 --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_SIMCLR_HPA --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_SIMCLR_JUMP --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_SIMCLR_SIM --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_SIMCLR_STED --seed $seed --blocks "all" #--overwrite
        # python finetune_v2.py --dataset $dataset --model resnet18 --weights RESNET18_DINO_STED --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --seed $seed --from-scratch #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_IMAGENET1K_V1 --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SIMCLR_HPA --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SIMCLR_JUMP --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SIMCLR_SIM --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet50 --weights RESNET50_SIMCLR_STED --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --seed $seed --from-scratch #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_IMAGENET1K_V1 --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SIMCLR_HPA --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SIMCLR_JUMP --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SIMCLR_SIM --seed $seed --blocks "all" #--overwrite
        python finetune_v2.py --dataset $dataset --model resnet101 --weights RESNET101_SIMCLR_STED --seed $seed --blocks "all" #--overwrite
    done
done