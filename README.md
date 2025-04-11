# STED-FM
Repository for the paper `A Foundation Model for Super-Resolution Microscopy Enabling Multi-Task Analysis, Representation-Based Discovery, and Interactive Microscopy`.

## Installation 
```
git clone https://github.com/FLClab/flc-dataset.git
cd flc-dataset
pip install -e .
```

## Example usage
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

## Download models

To download the models use the following
```bash
mkdir -p "${HOME}/.stedfm"
rclone copy --progress "valeria-s3:flclab-foundation-models/models/mae-small-sted.zip" "${HOME}/.stedfm"
unzip "${HOME}/.stedfm/mae-small-sted.zip" -d "${HOME}/.stedfm"
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

