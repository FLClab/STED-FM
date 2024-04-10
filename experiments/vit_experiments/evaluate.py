import torch
from timm.models.vision_transformer import vit_small_patch16_224
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import lightly.models.utils
import argparse
import numpy as np
from tqdm import tqdm
from models.classifier import MAEClassificationHead
import sys
sys.path.insert(0, "../../proteins_experiments")
from utils.data_utils import fewshot_loader

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='finetuned')
args = parser.parse_args()



class LightlyMAE(torch.nn.Module):
    def __init__(self, vit) -> None:
        super().__init__()
        decoder_dim = 512
        self.mask_ratio = 0.0
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
    
def compute_Nary_accuracy(preds: torch.Tensor, labels: torch.Tensor, N: int = 4) -> list:
    # accuracies = []
    correct = []
    big_n = []
    _, preds = torch.max(preds, 1)
    assert preds.shape == labels.shape
    c = torch.sum(preds == labels)
    correct.append(c.item())
    big_n.append(preds.shape[0])
    for n in range(N):
        c = ((preds == labels ) * (labels == n)).float().sum().cpu().detach().numpy()
        n = (labels==n).float().sum().cpu().detach().numpy()
        correct.append(c)
        big_n.append(n)
        # temp = ( (preds == labels) * (labels == n)).float().sum() / (labels == n).float().sum()
        # accuracies.append(temp.cpu().detach().numpy())
    return np.array(correct), np.array(big_n)


def load_model():
    vit = vit_small_patch16_224(in_chans=1)
    backbone = LightlyMAE(vit=vit)
    model = MAEClassificationHead(
        backbone=backbone,
        feature_dim=384,
        num_classes=4,
        freeze=args.freeze,
        global_pool="avg"
    )
    checkpoint = torch.load(f"../Datasets/FLCDataset/baselines/{args.model}_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate(
        model,
        loader, 
        device
):
    big_correct = np.array([0] * (4+1))
    big_n = np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, proteins, conditions in tqdm(loader, desc="Evaluation..."):
            labels = proteins if args.class_type == 'protein' else conditions
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions = model(imgs)
            correct, n = compute_Nary_accuracy(predictions, labels)
            big_correct = big_correct + correct
            big_n = big_n + n
        accuracies = big_correct / big_n
        print("********* Validation metrics **********")
        print("Overall accuracy = {:.3f}".format(accuracies[0]))
        for i in range(1, 4+1):
            acc = accuracies[i]
            print("Class {} accuracy = {:.3f}".format(
                i, acc))


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")
    model = load_model().to(DEVICE)
    model.eval()
    _, _, loader = fewshot_loader(
        path=args.datapath,
        class_type=args.class_type,
        n_channels=1,
    )
    evaluate(model=model, loader=loader, device=DEVICE)

if __name__=="__main__":
    main()