import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
import lightly.models.utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import argparse 
import random
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np
# All imports before ones from my own packages should come before this line
sys.path.insert(0, "../../proteins_experiments")
from utils.data_utils import load_theresa_proteins


parser = argparse.ArgumentParser()
parser.add_argument("--class-type", "-ct", type=str, default='protein')
parser.add_argument("--pretraining", type=str, default='lightly')
parser.add_argument("--datapath", type=str, default="")
args = parser.parse_args()

class LightlyMAE(torch.nn.Module):
    def __init__(self, vit) -> None:
        super().__init__()
        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            in_chans=1,
            decoder_depth=1,
            decoder_num_heads=8,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward_encoder(self, images: torch.Tensor, idx_keep: bool=None) -> torch.Tensor:
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x: torch.Tensor, idx_keep: bool, idx_mask: bool) -> torch.Tensor:
        batch_size = x.shape[0]
        x_decode = self.decoder.embed(x)
        x_masked = lightly.models.utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = lightly.models.utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))
        x_decoded = self.decoder.decode(x_masked)
        x_pred = lightly.models.utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        idx_keep, idx_mask = lightly.models.utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
        patches = lightly.models.utils.patchify(images, self.patch_size)
        target = lightly.models.utils.get_at_index(patches, idx_mask-1)
        return x_pred, target
    
def load_model():
    vit = vit_small_patch16_224(in_chans=1)
    model = LightlyMAE(vit=vit)
    checkpoint = torch.load("/home/frederic/Datasets/FLCDataset/baselines/MAE/checkpoint-190.pth")
    model.load_state_dict(checkpoint['model'])
    return model

def knn_predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    out = defaultdict(list)
    with torch.no_grad():
        for x, proteins, conditions in tqdm(loader, desc="Extracting features..."):
            labels = proteins if args.class_type == "protein" else conditions
            x, labels = x.to(device), labels.to(device)
            features = model.forward_encoder(x)
            out['features'].extend(features.cpu().data.numpy())
            out['labels'].extend(labels.cpu().data.numpy().tolist())
    samples = np.array(out['features'])
    labels = np.array([int(item) for item in out['labels']])
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(samples)
    neighbors = neigh.kneighbors(samples, return_distance=False)[:, 1:]
    associated_labels = labels[neighbors]

    uniques = np.unique(labels)
    print(np.unique(labels, return_counts=True))
    confusion_matrix = np.zeros((len(uniques), len(uniques)))
    for unique in tqdm(uniques, desc="Confusion matrix computation..."):
        mask = labels == unique
        for predicted_unique in uniques:
            votes = np.sum((associated_labels[mask] == predicted_unique).astype(int), axis=-1)
            confusion_matrix[unique, predicted_unique] += np.sum(votes >= 3)
    accuracy = np.diag(confusion_matrix).sum() / np.sum(confusion_matrix)
    print(f"Accuracy: {accuracy * 100:0.2f}")
    acc = accuracy * 100
    fig, ax = plt.subplots()
    cm = confusion_matrix / np.sum(confusion_matrix, axis=-1)[np.newaxis]
    ax.imshow(cm, vmin=0, vmax=1, cmap="Purples")
    for j in range(cm.shape[-2]):
        for i in range(cm.shape[-1]):
            ax.annotate(
                f"{cm[j, i]:0.2f}\n({confusion_matrix[j, i]:0.0f})", (i, j), 
                horizontalalignment="center", verticalalignment="center",
                color="white" if cm[j, i] > 0.5 else "black"
            )
    ax.set(
        xticks=uniques, yticks=uniques,  
    )
    ax.set_title(round(acc, 4))
    fig.savefig(f"./results/{args.pretraining}/{args.class_type}_knn_results.pdf", dpi=1200, bbox_inches='tight', transparent=True)
    plt.close(fig)

def main():
    feature_dims = 384 # MAE's ViT embedding dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    loader = load_theresa_proteins(
        path="/home/frederic/Datasets/FLCDataset",
        class_type=args.class_type,
        n_channels=1,
    )
    model = load_model().to(device)
    model.eval()
    knn_predict(model=model, loader=loader, device=device)
    

if __name__=="__main__":
    main()