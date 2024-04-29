############################################################################################################
############################################################################################################
# FACTIN
############################################################################################################
############################################################################################################

DATASET="factin"

############################################################################################################
# MICRANET
############################################################################################################

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/from-scratch \
#     --dataset ${DATASET} \
#     --backbone micranet \
#     --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/pretrained-MICRANET_SSL_STED \
#     --dataset ${DATASET} \
#     --backbone micranet \
#     --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/pretrained-frozen-MICRANET_SSL_STED \
#     --dataset ${DATASET} \
#     --backbone micranet \
#     --opts batch_size 32

############################################################################################################
# RESNET18
############################################################################################################

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/from-scratch/ \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-RESNET18_SSL_STED/ \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_SSL_STED \
#     --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-frozen-RESNET18_SSL_STED/ \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_SSL_STED \
#     --opts batch_size 32    

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-RESNET18_IMAGENET1K_V1/ \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_IMAGENET1K_V1 \
#     --opts batch_size 32    

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-frozen-RESNET18_IMAGENET1K_V1/ \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_IMAGENET1K_V1 \
#     --opts batch_size 32 

############################################################################################################
# RESNET50
############################################################################################################

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/from-scratch/ \
#     --dataset ${DATASET} \
#     --backbone resnet50 \
#     --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/pretrained-RESNET50_SSL_STED/ \
#     --dataset ${DATASET} \
#     --backbone resnet50 \
#     --backbone-weights RESNET50_SSL_STED \
#     --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/pretrained-frozen-RESNET50_SSL_STED/ \
#     --dataset ${DATASET} \
#     --backbone resnet50 \
#     --backbone-weights RESNET50_SSL_STED \
#     --opts batch_size 32    

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/pretrained-RESNET50_IMAGENET1K_V1/ \
#     --dataset ${DATASET} \
#     --backbone resnet50 \
#     --backbone-weights RESNET50_IMAGENET1K_V1 \
#     --opts batch_size 32    

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/pretrained-frozen-RESNET50_IMAGENET1K_V1/ \
#     --dataset ${DATASET} \
#     --backbone resnet50 \
#     --backbone-weights RESNET50_IMAGENET1K_V1 \
#     --opts batch_size 32     

############################################################################################################
############################################################################################################
# FOOTPROCESS
############################################################################################################
############################################################################################################

DATASET="footprocess"

############################################################################################################
# MICRANET
############################################################################################################

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/from-scratch \
#     --dataset ${DATASET} \
#     --backbone micranet \
#     --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/pretrained-MICRANET_SSL_STED \
#     --dataset ${DATASET} \
#     --backbone micranet \
#     --opts batch_size 32

# python eval-per-percentage.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/pretrained-frozen-MICRANET_SSL_STED \
#     --dataset ${DATASET} \
#     --backbone micranet \
#     --opts batch_size 32

############################################################################################################
# RESNET18
############################################################################################################

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/from-scratch/ \
    --dataset ${DATASET} \
    --backbone resnet18 \
    --opts batch_size 32

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-RESNET18_SSL_STED/ \
    --dataset ${DATASET} \
    --backbone resnet18 \
    --backbone-weights RESNET18_SSL_STED \
    --opts batch_size 32

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-frozen-RESNET18_SSL_STED/ \
    --dataset ${DATASET} \
    --backbone resnet18 \
    --backbone-weights RESNET18_SSL_STED \
    --opts batch_size 32    

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-RESNET18_IMAGENET1K_V1/ \
    --dataset ${DATASET} \
    --backbone resnet18 \
    --backbone-weights RESNET18_IMAGENET1K_V1 \
    --opts batch_size 32    

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-frozen-RESNET18_IMAGENET1K_V1/ \
    --dataset ${DATASET} \
    --backbone resnet18 \
    --backbone-weights RESNET18_IMAGENET1K_V1 \
    --opts batch_size 32 

############################################################################################################
# RESNET50
############################################################################################################

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/from-scratch/ \
    --dataset ${DATASET} \
    --backbone resnet50 \
    --opts batch_size 32

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/pretrained-RESNET50_SSL_STED/ \
    --dataset ${DATASET} \
    --backbone resnet50 \
    --backbone-weights RESNET50_SSL_STED \
    --opts batch_size 32

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/pretrained-frozen-RESNET50_SSL_STED/ \
    --dataset ${DATASET} \
    --backbone resnet50 \
    --backbone-weights RESNET50_SSL_STED \
    --opts batch_size 32    

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/pretrained-RESNET50_IMAGENET1K_V1/ \
    --dataset ${DATASET} \
    --backbone resnet50 \
    --backbone-weights RESNET50_IMAGENET1K_V1 \
    --opts batch_size 32    

python eval-per-percentage.py \
    --restore-from ./data/SSL/segmentation-baselines/resnet50/${DATASET}/pretrained-frozen-RESNET50_IMAGENET1K_V1/ \
    --dataset ${DATASET} \
    --backbone resnet50 \
    --backbone-weights RESNET50_IMAGENET1K_V1 \
    --opts batch_size 32         