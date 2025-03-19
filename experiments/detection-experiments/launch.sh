WEIGHTS=(
    "MAE_SMALL_STED"
    "MAE_SMALL_SIM"
    "MAE_SMALL_HPA"
    "MAE_SMALL_JUMP"
    "MAE_SMALL_IMAGENET1K_V1"    
)
TEMPLATES=(
    "./templates/template-factin-assemblies-jchabbert.tif"
    "./templates/template-factin-spots-jchabbert.tif"
    "./templates/template-factin-spine-jchabbert.tif"
)
for template in "${TEMPLATES[@]}"
do
    for weight in "${WEIGHTS[@]}"
    do
        python main-from-template-image.py \
                --backbone-weights "${weight}" \
                --dataset "/home-local2/projects/SSL/detection-data/multistructure-factin/" \
                --template "${template}"
    done
done