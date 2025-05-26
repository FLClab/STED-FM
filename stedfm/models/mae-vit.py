import torch 
import random
import os
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from multiprocessing import Manager
from lightning.pytorch.core import LightningDataModule
import sys
import torch.distributed
from stedfm.models.custom_vit import build_mae_lightning_64_p8
from stedfm.models.lightly_mae import get_backbone
from stedfm import get_pretrained_model_v2
import torchvision.transforms as T
from stedfm.datasets import get_dataset
from stedfm.models import get_model
from stedfm.model_builder import get_base_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )


dataset = get_dataset("synaptic-proteins",
    path= "/home-local/Tassnym/Datasets/LargeProteinModels/synaptic_proteins_train_catalog.tar",
    use_cache=False,                
    max_cache_size=16e9,            
    transform=None,                
    crop_size=64,                   
    anomaly_prob=0.0                
)

model,cfg = get_base_model(name="mae-lightning-64-p8", crop_size = 64)
model = model.to(device)
model.eval()

print(model)
print(f"Model device: {next(model.parameters()).device}")

dataloader = DataLoader(dataset, batch_size = 4, shuffle=True)

with torch.no_grad():
    for crop, mask_crop, latent_vector, metadata in dataloader:
        crop = crop.to(device)
        output = latent_vector.to(device)
        print("Input crop shape:", crop.shape)
        print("Output shape:", output.shape)
        break

