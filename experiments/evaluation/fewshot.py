import numpy as np
import matplotlib.pyplot as plt 
import torch
import argparse 
from tqdm import tqdm 
from torch.optim.lr_scheduler import CosineAnnealingLR 
import sys 
sys.path.insert(0, "../")
from loaders import get_dataset 
from model_builder import get_pretrained_model_v2 
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='synaptic-proteins')
parser.add_argument("--model", type=str, default='mae-small')
parser.add_argument("--weights", type=str, default="")
parser.add_argument("--frozen-blocks", type=str, default='0')
parser.add_argument("--label-percentage", type=float, 0.01)
args = parser.parse_args()


def train():
    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if "imagenet" in args.weights.lower() else 1
    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if "imagenet" in args.weights.lower() else 1,
        in_channels=n_channels,
        as_classifier=True,
        blocks=args.frozen_blocks,
    )
    model = model.to(device)
    batch_size=cfg.batch_size 
    train_loader, valid_loader, _ = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        batch_size=batch_size,
        training=True,
        fewshot_pct=args.label_percentage
    )
    
    optimizer= torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.99))

    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100)
    criterion = torch.nn.CrossEntropyLoss()


    

if __name__=="__main__":
    main()