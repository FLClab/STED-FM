import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import tarfile
from torch.utils.data import Dataset, DataLoader
import argparse 
from typing import Optional, Callable, Tuple
import io
from tqdm import tqdm
from timm.models.layers import PatchEmbed 
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import sys 
import torch.nn.functional as F
import os
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default="/home-local/Frederic/Datasets/Neural-Activity-States/PSD95-Basson/synaptic-protein-segmentation.tar")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
args = parser.parse_args()

class ProteinSegmentationDataset(Dataset):
    def __init__(
        self,
        archive_path: str,
        transform: Optional[Callable] = None,
        n_channels: int = 1,
    ) -> None:
        self.archive_path = archive_path
        self.transform = transform
        self.n_channels = n_channels
        self.archive_obj = tarfile.open(self.archive_path, "r")
        self.members = list(sorted(self.archive_obj.getmembers(), key=lambda m: m.name))

    def __len__(self) -> int:
        return len(self.members)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        member = self.members[idx]
        buffer = io.BytesIO()
        buffer.write(self.archive_obj.extractfile(member).read())
        buffer.seek(0)
        data = np.load(buffer, allow_pickle=True)
        data = {key : values for key, values in data.items()}
        img, mask = data["img"], data["segmentation"]
        img = torch.tensor(img[np.newaxis, ...], dtype=torch.float32)
        mask = torch.tensor(mask[np.newaxis, ...], dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")
    dataset = ProteinSegmentationDataset(
        archive_path=args.datapath,
        transform=None,
        n_channels=1,
    )
    
    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=False,
        in_channels=1,
        as_classifier=True,
        blocks="all",
        num_classes=2
    )
    model.to(DEVICE)
    
    nodes, _ = get_graph_node_names(model.backbone, tracer_kwargs={'leaf_modules': [PatchEmbed]})
    block_num = 11  
    for i in range(len(dataset)):
        img, mask = dataset[i]
        img = img.to(DEVICE).unsqueeze(0)
        mask = mask.to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            feature_extractor = create_feature_extractor( 
                model, return_nodes=[f'backbone.blocks.{block_num}.attn.q_norm', f'backbone.blocks.{block_num}.attn.k_norm'],
                tracer_kwargs={'leaf_modules': [PatchEmbed]})
            out = feature_extractor(img)
            q, k = out[f'backbone.blocks.{block_num}.attn.q_norm'], out[f'backbone.blocks.{block_num}.attn.k_norm']
            factor = (384 / 6) ** -0.5 # (head_dim / num_heads ) ** -0.5
            q = q * factor 
            attn = q @ k.transpose(-2, -1) # (1, 6, 197, 197)
            attn = attn.softmax(dim=-1) # (1, 6, 197, 197)
            attn_map = attn.mean(dim=1).squeeze(0)  # (197, 197)
            cls_attn_map = attn[:, :, 0, 1:]
            cls_attn_map = cls_attn_map.mean(dim=1).view(14, 14).detach() 
            cls_resized = F.interpolate(cls_attn_map.view(1, 1, 14, 14), (224, 224), mode='bilinear').view(224, 224, 1)
            cls_resized = cls_resized.squeeze().cpu().detach().numpy()
            tau = np.quantile(cls_resized, 0.95)
            cls_resized = cls_resized > tau
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            axs[0].imshow(img.squeeze().cpu().detach().numpy(), cmap="hot", vmin=0, vmax=1)
            axs[1].imshow(mask.squeeze().cpu().detach().numpy(), cmap="gray")
            axs[2].imshow(cls_resized, cmap="gray")
            fig.savefig(f"./img_{i}.png", bbox_inches="tight", dpi=1200)
            plt.close(fig)


    