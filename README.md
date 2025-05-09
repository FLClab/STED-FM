# STED-FM
Repository for the paper `A Foundation Model for Super-Resolution Microscopy Enabling Multi-Task Analysis, Representation-Based Discovery, and Interactive Microscopy`.

## Installation 
```
git clone https://github.com/FLClab/flc-dataset.git
cd flc-dataset
pip install -e .
```

## Example usage of the backbone
```python
from stedfm import get_pretrained_model_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, cfg = get_pretrained_model_v2(
    name="mae-lightning-small",
    weights="MAE_SMALL_STED",
    as_classifier=True,
    # global_pool="patch"
)
model.to(device)
model.eval()
with torch.no_grad():
    img = torch.randn(1, 1, 224, 224).to(device)
    out = model.forward_features(img) # (1, 384) --> uncomment the global_pool line to return all embeddings (1, 196, 384)
```

## Example usage of adding a decoder to the backbone 
```python
from stedfm import get_pretrained_model_v2, get_decoders
from stedfm.configuration import Configuration

class SegmentationConfiguration(Configuration):
    
    freeze_backbone: bool = True 
    num_epochs: int = 300
    learning_rate: float = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone, cfg = get_pretrained_model_v2(
    name="mae-lightning-small",
    weights="MAE_SMALL_STED",
)

model = get_decoder(backbone, cfg).to(device)
with torch.no_grad():
    img = torch.randn(1, 1, 224, 224).to(device)
    out = model(img)
```


## Folder Architecture

Here's the folder architecture that is assumed in the repository...
```bash
baselines
|--- resnet18
segmentation-baselines
|--- resnet18
|---|--- <DATASET>
ssl-data
|--- <DATASET>
evaluation-data
|--- <DATASET>
segmentation-data
|--- <DATASET>
```

