import torch
import matplotlib.pyplot as plt 
import numpy as np
import argparse
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim
from datasets import get_dataset 
from pytorch_msssim import ms_ssim
import os 
import random
import sys
sys.path.insert(0, "../")
from model_builder import get_pretrained_model_v2
from utils import SaveBestModel, AverageMeter
sys.path.insert(1, '../segmentation-experiments')
from decoders.vit import get_decoder


parser = argparse.ArgumentParser()
parser.add_argument("--restore-from", type=str, default="")
parser.add_argument("--dataset", type=str, default="synaptic-denoising")
parser.add_argument("--backbone", type=str, default='mae-small')
parser.add_argument("--backbone-weights", type=str, default="MAE_SSL_STED")
args = parser.parse_args()

def get_save_folder() -> str:
    if "imagenet" in args.backbone_weights.lower():
        return "ImageNet"
    elif "sted" in args.backbone_weights.lower():
        return "STED"
    elif "jump" in args.backbone_weights.lower():
        return "JUMP"
    else:
        return "CTC"
    
SAVE_NAME = get_save_folder()
    
def denoising_loss(predictions: torch.Tensor, labels: torch.Tensor, sim_weight: float = 1.0):
    recon_loss = torch.nn.functional.mse_loss(predictions, labels)
    sim_loss = 1 - ms_ssim(predictions, labels, data_range=1.0, size_average=True)
    return recon_loss + (sim_weight * sim_loss)

def compute_results(model, loader, device):
    losses = []
    img_data = np.zeros((200, 224, 224))
    label_data = np.zeros((200, 224, 224))
    pred_data = np.zeros((200, 224, 224))
    counter = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Evaluation...'):
            imgs, labels = imgs.to(device), labels.to(device)
            predictions = model(imgs)
            loss = denoising_loss(predictions, labels)
            losses.append(loss.item())
            for img, l, pred in zip(imgs, labels, predictions):
                prob = random.random()
                if prob > 0.5 and counter < 200:
                    img = torch.permute(img, dims=(1,2,0)).squeeze().cpu().detach().numpy()
                    if SAVE_NAME == "ImageNet":
                        img = np.squeeze(img[:, :, 0])

                    l = torch.permute(l, dims=(1,2,0)).squeeze().cpu().detach().numpy()
                    pred = torch.permute(pred, dims=(1,2,0)).squeeze().cpu().detach().numpy()
                    img_data[counter] = img
                    label_data[counter] = l
                    pred_data[counter] = pred
                    counter += 1
    np.savez(f"./results/inferred/{args.backbone}_{SAVE_NAME}_{args.dataset}", imgs=img_data, labels=label_data, predictions=pred_data, losses=losses)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3 if SAVE_NAME == "ImageNet" else 1

    backbone, cfg = get_pretrained_model_v2(
        name=args.backbone,
        weights=args.backbone_weights,
        as_classifier=False,
        blocks='all',
        path=None,
        mask_ratio=0.0,
        pretrained=True if SAVE_NAME == "ImageNet" else 1,
        in_channels=n_channels
    )
    cfg.freeze_backbone = True


    cfg.batch_size = 1 # To go image by image
    _, _, test_dataset = get_dataset(
        name=args.dataset,
        cfg=cfg, 
        n_channels=n_channels
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=6,
    )

    checkpoint = torch.load(args.restore_from)
    print(checkpoint.keys())
    OUTPUT_FOLDER = os.path.dirname(args.restore_from)
    model = get_decoder(backbone, cfg, in_channels=n_channels, out_channels=1)
    ckpt = checkpoint.get("model_state_dict", None)
    if not ckpt is None:
        print(f"--- Restoring model {args.backbone} | {args.backbone_weights} ---")
        model.load_state_dict(ckpt)
    else:
        raise ValueError
    model = model.to(DEVICE)

    model.eval()

    compute_results(model=model, loader=test_loader, device=DEVICE)



if __name__=="__main__":
    main()