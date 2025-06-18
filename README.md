# STED-FM
Repository for the paper `A Self-Supervised Foundation Model for Robust and Generalizable Representation Learning in STED Microscopy`.

## Abstract
Foundation Models (FMs) have drastically increased the potential and power of deep learning algorithms in the fields of natural language processing and computer vision. However, their application in specialized fields like biomedical imaging, and fluorescence microscopy remains difficult due to distribution shifts, and the scarcity of high-quality annotated datasets. The high cost of data acquisition and the requirement for in-domain expertise further exacerbate this challenge in super-resolution microscopy. To address this, we introduce STED-FM, a foundation model specifically designed for super-resolution STimulated Emission Depletion (STED) microscopy. STED-FM leverages a Vision Transformer (ViT) architecture trained at scale with Masked Autoencoding (MAE) on a new dataset of nearly one million STED images. STED-FM learns expressive latent representations without extensive annotations. Our comprehensive evaluation demonstrates STED-FM's versatility across diverse downstream tasks. Unsupervised experiments highlight the discriminative nature of its learned latent space. Our model significantly reduces the need for annotated data required to achieve strong performance in classification and segmentation tasks, both in- and out-of-distribution. Furthermore, STED-FM enhances diffusion model-generated images, enabling latent attribute manipulation and the discovery of novel and subtle nanostructures and phenotypes. Its structure retrieval capabilities are also integrated into automated STED microscopy acquisition pipelines. The high expressivity and strong performance across tasks make STED-FM a compelling resource for researchers analyzing super-resolution STED images.

## System Requirements

### Hardware requirements
Training of the STED-FM model most likely requires a multi-GPU setup for reasonable run time. In our case, it was trained on a high-performance computing system with the following specifications:
- 1 node with 10 CPU core
- 4 Tesla V100-SXM2-16Gb GPU per node
- Allocation of the entire node's memory (maximum of 191Gb)

The time required to train the models depends on the dataset size and the number of epochs. With the specifications listed above, training of the ViT-small backbone took 24 hours. To maintain a 24-hour training for the ViT-large backbone, the number of nodes needed to be increased to 4 (i.e., 16 GPUs).  

After pre-training, all fine-tuning experiments of the STED-FM model can be run on a single-GPU setup. The longer configuration is the fine-tuning of the full network, which in some settings can take up to 12 hours.

### Software requirements. 
*OS requirements*
The source code was test on the Linux - CentOS 7 operating system.

*Python dependencies*
The code was run using Python 3.8.18, 3.10.13 and 3.11.5. The required python installation should be `python>=3.8`.

## Installation 
```
git clone https://github.com/FLClab/STED-FM.git
cd STED-FM
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

## Download models

To download the models use the following
```bash
mkdir -p "${HOME}/.stedfm"
rclone copy --progress "valeria-s3:flclab-foundation-models/models/mae-small-sted.zip" "${HOME}/.stedfm"
unzip "${HOME}/.stedfm/mae-small-sted.zip" -d "${HOME}/.stedfm"
```

## Project Structure & Experiments

The training of STED-FM is done through the `pretrain_lightning.py` script. 
The repository is mostly split in sub-folders of the `experiments` folder, corresponding to the various families of tasks that were performed by the model in the paper. A separate README is provided for every set of experiments.
- The unsupervised experiments (recursive clustering, image retrieval) are in the `clustering-experiments` sub-folder.
- The supervised classification fine-tuning experiments as well as KNN classification experiment are in the `evaluation` sub-folder. 
- The supervised segmentation experiments as well as the patch retrieval experiment can be found in the `segmentation-experiments` sub-folder. 
- The image generation and latent attribute manipulation experiments are in the `diffusion-experiments` sub-folder.
- The code for extracting attention maps from STED-FM is in the `interpretability-experiments` sub-folder.
- The code for performing user studies on the various tasks are in the `user-study` sub-folder.

## Documentation
Below is an example command for pre-training the ViT-S architecture with the large-scale STED dataset, which results in STED-FM. Note that this will probably be infeasibly long with a single GPU.

```bash
python pretrain_lightning.py --seed 42 --model mae-lightning-small --dataset STED --use-tensorboard --save-folder <path/to/save/checkpoint> --dataset-path "<path/to/STED/dataset/stedfm-dataset-crops.tar>"
```

## Citation



