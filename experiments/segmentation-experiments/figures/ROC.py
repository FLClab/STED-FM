import torch 
import torchvision 
import numpy as np 
import os 
from typing import Tuple, List, Dict, Optional 
from tqdm import tqdm 
from collections import defaultdict 
from stedfm import get_decoder, get_pretrained_model_v2 
import copy 
from stedfm.configuration import Configuration  
from stedfm.DEFAULTS import BASE_PATH, COLORS
import argparse 
import glob 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve
import sys 
sys.path.insert(0, "../")
from datasets import get_dataset

# Define common FPR points for interpolation
MEAN_FPR = np.linspace(0.01, 0.99, 100) 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small") 
parser.add_argument("--mode", type=str, default="pretrained-frozen")
parser.add_argument("--dataset", type=str, default="factin")
args = parser.parse_args()

class SegmentationConfiguration(Configuration):
    
    freeze_backbone: bool = True
    num_epochs: int = 300
    learning_rate: float = 1e-4

def compute_roc(pred, y):
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.squeeze().cpu().numpy() 

    tpr_per_class = defaultdict(list)
   
    for ch in range(pred.shape[0]):
        pred_ch, y_ch = pred[ch], y[ch]
        pred_flat, y_flat = pred_ch.ravel(), y_ch.ravel()
        if args.dataset == "synaptic-semantic-segmentation":
            y_flat = y_flat > 0.5 # majority voting
            
        if not np.any(y_flat) and not np.any(pred_flat):
            # Append array of -1s with same shape as interpolated TPR
            tpr_per_class[ch].append(np.full_like(MEAN_FPR, -1).tolist())
            continue
        if np.unique(y_flat).size == 1:
            # Append array of -1s with same shape as interpolated TPR
            tpr_per_class[ch].append(np.full_like(MEAN_FPR, -1).tolist())
            continue
        fpr, tpr, _ = roc_curve(y_true=y_flat, y_score=pred_flat)
        
        # Interpolate TPR to common FPR points
        interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure ROC curve starts at origin
        
        tpr_per_class[ch].append(interp_tpr.tolist())
    return tpr_per_class

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    backbone_weights = [None, "MAE_SMALL_IMAGENET1K_V1", "MAE_SMALL_JUMP", "MAE_SMALL_HPA", "MAE_SMALL_SIM", "MAE_SMALL_STED"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for weights in backbone_weights:
        backbone, cfg = get_pretrained_model_v2(
            name=args.model,
            weights=weights,
        )
        _, _, test_dataset = get_dataset(
            name=args.dataset,
            cfg=cfg,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=False,
            drop_last=False,
        )
        segmentation_cfg = SegmentationConfiguration() 
        for key, value in segmentation_cfg.__dict__.items():
            setattr(cfg, key, value)
        cfg.backbone_weights = weights 
        model = get_decoder(backbone, cfg)

        if weights is None:
            model_paths = glob.glob(f"{BASE_PATH}/segmentation-baselines/{args.model}/{args.dataset}/from-scratch-*/result.pt", recursive=True)
            model_paths = [p for p in model_paths if "labels" not in p]
            print(f"Found {len(model_paths)} model paths for {args.mode} mode")
        else:
            model_paths = glob.glob(f"{BASE_PATH}/segmentation-baselines/{args.model}/{args.dataset}/{args.mode}-{weights}-*/result.pt", recursive=True)
            model_paths = [p for p in model_paths if "labels" not in p]
            if args.mode == "pretrained":
                model_paths = [p for p in model_paths if "frozen" not in p]

        avg_tpr = defaultdict(list)
        if weights is None:
            weights = "from-scratch"
        for path in model_paths:
            tprs = defaultdict(list)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                for X, y in tqdm(test_loader, desc=f"Processing {weights}"):
                    X, y = X.to(DEVICE), y.to(DEVICE) 
                    pred = model(X) 
                    tpr_per_class = compute_roc(pred, y)
                    for key in tpr_per_class.keys():
                        # tpr_per_class[key] is a list with one element per batch
                        tprs[key].extend(tpr_per_class[key])

            for key in tprs.keys():
                tp = np.array(tprs[key])
                tp_masked = np.ma.masked_equal(tp, -1)
                tp_mean = np.ma.mean(tp_masked, axis=0)
                avg_tpr[key].append(tp_mean)

        final_tpr = [] 
        for key in avg_tpr.keys():
            tpr_mean = np.mean(avg_tpr[key], axis=0)
            final_tpr.append(tpr_mean)

        final_tpr = np.array(final_tpr)
        tpr_mean = np.mean(final_tpr, axis=0)
        tpr_err = np.std(final_tpr, axis=0)

        # Plot using MEAN_FPR (common FPR points) and computed TPR
        ax.plot(MEAN_FPR, tpr_mean, label=weights, color=COLORS[weights])
        ax.fill_between(MEAN_FPR, tpr_mean - tpr_err, tpr_mean + tpr_err, alpha=0.2, color=COLORS[weights])

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.savefig(f"./results/{args.dataset}-{args.mode}-ROC-curve.pdf", bbox_inches="tight", dpi=800)
    plt.close(fig)


        
if __name__ == "__main__":
    main()