
MODELS=(
    "resnet18_STED"
    "resnet18_HPA"
    "resnet18_JUMP"
)

for model in ${MODELS[@]}
do
  	rsync -avz --progress --no-g --no-p \
                --include="result.pt" --exclude="*" \
                "/home/anbil106/scratch/projects/SSL/baselines/dataset-fullimages-1Msteps-multigpu/${model}/" \
                "${HOME}/projects/def-flavielc/baselines/${model}/"
done