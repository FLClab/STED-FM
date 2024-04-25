############################################################################################################
############################################################################################################
# FACTIN
############################################################################################################
############################################################################################################

############################################################################################################
# RESNET18
############################################################################################################

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/factin/from-scratch/result.pt \
#     --dataset factin \
#     --backbone resnet18 \
#     --opts batch_size 32

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/factin/pretrained-RESNET18_SSL_STED/result.pt \
#     --dataset factin \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_SSL_STED \
#     --opts batch_size 32

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/factin/pretrained-frozen-RESNET18_SSL_STED/result.pt \
#     --dataset factin \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_SSL_STED \
#     --opts batch_size 32    

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/factin/pretrained-RESNET18_IMAGENET1K_V1/result.pt \
#     --dataset factin \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_IMAGENET1K_V1 \
#     --opts batch_size 32    

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/factin/pretrained-frozen-RESNET18_IMAGENET1K_V1/result.pt \
#     --dataset factin \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_IMAGENET1K_V1 \
#     --opts batch_size 32 

############################################################################################################
# RESNET50
############################################################################################################

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/factin/from-scratch/result.pt \
#     --dataset factin \
#     --backbone resnet50 \
#     --opts batch_size 32

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/factin/pretrained-RESNET50_SSL_STED/result.pt \
#     --dataset factin \
#     --backbone resnet50 \
#     --backbone-weights RESNET50_SSL_STED \
#     --opts batch_size 32

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/factin/pretrained-frozen-RESNET50_SSL_STED/result.pt \
#     --dataset factin \
#     --backbone resnet50 \
#     --backbone-weights RESNET50_SSL_STED \
#     --opts batch_size 32    

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/factWHatin/pretrained-RESNET50_IMAGENET1K_V1/result.pt \
#     --dataset factin \
#     --backbone resnet50 \
#     --backbone-weights RESNET50_IMAGENET1K_V1 \
#     --opts batch_size 32    

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet50/factin/pretrained-frozen-RESNET50_IMAGENET1K_V1/result.pt \
#     --dataset factin \
#     --backbone resnet50 \
#     --backbone-weights RESNET50_IMAGENET1K_V1 \
#     --opts batch_size 32 

############################################################################################################
# MICRANET
############################################################################################################

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/factin/from-scratch/result.pt \
#     --dataset factin \
#     --backbone micranet \
#     --opts batch_size 32

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/factin/pretrained-MICRANET_SSL_STED/result.pt \
#     --dataset factin \
#     --backbone micranet \
#     --backbone-weights MICRANET_SSL_STED \
#     --opts batch_size 32

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/micranet/factin/pretrained-frozen-MICRANET_SSL_STED/result.pt \
#     --dataset factin \
#     --backbone micranet \
#     --backbone-weights MICRANET_SSL_STED \
#     --opts batch_size 32    

############################################################################################################
############################################################################################################
# FOOTPROCESS
############################################################################################################
############################################################################################################

DATASET="footprocess"

############################################################################################################
# RESNET18
############################################################################################################

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/from-scratch/result.pt \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --opts batch_size 32

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-RESNET18_SSL_STED/result.pt \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_SSL_STED \
#     --opts batch_size 32

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-frozen-RESNET18_SSL_STED/result.pt \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_SSL_STED \
#     --opts batch_size 32    

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-RESNET18_IMAGENET1K_V1/result.pt \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_IMAGENET1K_V1 \
#     --opts batch_size 32    

# python eval.py \
#     --restore-from ./data/SSL/segmentation-baselines/resnet18/${DATASET}/pretrained-frozen-RESNET18_IMAGENET1K_V1/result.pt \
#     --dataset ${DATASET} \
#     --backbone resnet18 \
#     --backbone-weights RESNET18_IMAGENET1K_V1 \
#     --opts batch_size 32 

############################################################################################################
# MICRANET
############################################################################################################

python eval.py \
    --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/from-scratch/result.pt \
    --dataset ${DATASET} \
    --backbone micranet \
    --opts batch_size 32

python eval.py \
    --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/pretrained-MICRANET_SSL_STED/result.pt \
    --dataset ${DATASET} \
    --backbone micranet \
    --backbone-weights MICRANET_SSL_STED \
    --opts batch_size 32

python eval.py \
    --restore-from ./data/SSL/segmentation-baselines/micranet/${DATASET}/pretrained-frozen-MICRANET_SSL_STED/result.pt \
    --dataset ${DATASET} \
    --backbone micranet \
    --backbone-weights MICRANET_SSL_STED \
    --opts batch_size 32        