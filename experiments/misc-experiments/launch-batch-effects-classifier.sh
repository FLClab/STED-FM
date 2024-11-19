WEIGHTS=(
    "ImageNet"
    "HPA"
    "JUMP"
    "STED"
)
# for weights in ${WEIGHTS[@]}; do
#     python main-batch-effects-classifier.py --model resnet18 --pretraining ${weights} --dataset optim --probe "linear-probe_None_42.pth"
# done
for weights in ${WEIGHTS[@]}; do
    python main-batch-effects-classifier.py --model mae-lightning-tiny --pretraining ${weights} --dataset optim
done