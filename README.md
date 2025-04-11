# STED-FM
Repository for the paper `A Foundation Model for Super-Resolution Microscopy Enabling Multi-Task Analysis, Representation-Based Discovery, and Interactive Microscopy`.

## Installation 
```
git clone https://github.com/FLClab/flc-dataset.git
cd flc-dataset
pip install -e .
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

