WEIGHTS=(
    # "MAE_SMALL_IMAGENET1K_V1"
    # "MAE_SMALL_JUMP"
    # "MAE_SMALL_HPA"
    "MAE_SMALL_STED"
)
for weights in ${WEIGHTS[@]}; do
    python main-batch-effects.py --model mae-lightning-small --weights ${weights} --dataset optim
done
# for weights in ${WEIGHTS[@]}; do
#     python main-batch-effects.py --model mae-lightning-small --weights ${weights} --dataset neural-activity-states
# done

# WEIGHTS=(
#     "RESNET18_IMAGENET1K_V1"
#     "RESNET18_SSL_JUMP"
#     "RESNET18_SSL_HPA"
#     "RESNET18_SSL_STED"
# )
# for weights in ${WEIGHTS[@]}; do
#     python main-batch-effects.py --model resnet18 --weights ${weights} --dataset optim
# done
# for weights in ${WEIGHTS[@]}; do
#     python main-batch-effects.py --model resnet18 --weights ${weights} --dataset neural-activity-states
# done