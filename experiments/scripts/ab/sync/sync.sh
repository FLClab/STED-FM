
MODELS=(
    "resnet18_STED"
    "resnet18_HPA"
    "resnet18_JUMP"
    "resnet50_STED"
)

for model in ${MODELS[@]}
do
  	rsync -avz --progress --no-g --no-p \
                --include="result.pt" --exclude="*" \
                "/home/anbil106/scratch/projects/SSL/baselines/dataset-fullimages-1Msteps-multigpu/${model}/" \
                "${HOME}/projects/def-flavielc/baselines/${model}/"
done

chmod g+wr -R "${HOME}/projects/def-flavielc/baselines/"