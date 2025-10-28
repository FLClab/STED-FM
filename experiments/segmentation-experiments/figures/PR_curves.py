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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
import sys 
sys.path.insert(0, "../")
from datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--metric", type=str, default="precision-recall")
parser.add_argument("--mode", type=str, default="pretrained-frozen")
parser.add_argument("--dataset", type=str, default="factin")
args = parser.parse_args()

class SegmentationConfiguration(Configuration):
    
    freeze_backbone: bool = True
    num_epochs: int = 300
    learning_rate: float = 1e-4

def compute_recall_precision(pred, y):
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.squeeze().cpu().numpy() 


    thresholds = np.linspace(0, 1, 100)
    recall_per_class = defaultdict(list)
    precision_per_class = defaultdict(list)
    for threshold in thresholds:
        for ch in range(pred.shape[0]):
            pred_ch, y_ch = pred[ch], y[ch]
            pred_binary = pred_ch > threshold 
            if not np.any(y_ch) and not np.any(pred_binary):
                recall_per_class[ch].append(-1)
                precision_per_class[ch].append(-1)
                continue
            if np.unique(y_ch).size == 1:
                recall_per_class[ch].append(-1)
                precision_per_class[ch].append(-1)
                continue 
            # Compute recall and precision for the current channel and threshold
            tp = np.logical_and(pred_binary, y_ch).sum()
            fp = np.logical_and(pred_binary, np.logical_not(y_ch)).sum()
            fn = np.logical_and(np.logical_not(pred_binary), y_ch).sum()
            
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            
            recall_per_class[ch].append(recall)
            precision_per_class[ch].append(precision)
    return recall_per_class, precision_per_class
            

def compute_weak_recall_precision(pred, y):
    pass
        


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_weights = ["MAE_SMALL_IMAGENET1K_V1", "MAE_SMALL_JUMP", "MAE_SMALL_HPA", "MAE_SMALL_SIM", "MAE_SMALL_STED"]

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
        model_paths = glob.glob(f"{BASE_PATH}/segmentation-baselines/{args.model}/{args.dataset}/{args.mode}-{weights}-*/result.pt", recursive=True)
        model_paths = [p for p in model_paths if "labels" not in p]
        if args.mode == "pretrained":
            model_paths = [p for p in model_paths if "frozen" not in p]
        model_paths = list(set(model_paths))
        for p in model_paths:
            print(p)

        average_recalls = defaultdict(list)
        average_precisions = defaultdict(list)
        for path in model_paths:
            recalls = defaultdict(list)
            precisions = defaultdict(list)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                for X, y in tqdm(test_loader, desc=f"Processing {weights}"):
                    X, y = X.to(DEVICE), y.to(DEVICE) 
                    pred = model(X)
                    recall_per_class, precision_per_class = compute_recall_precision(pred, y)
                    for key in recall_per_class.keys():
                        recalls[key].append(recall_per_class[key])
                        precisions[key].append(precision_per_class[key])

            for key in recalls.keys():
                r = np.array(recalls[key])
                p = np.array(precisions[key])
                r_masked = np.ma.masked_equal(r, -1)
                p_masked = np.ma.masked_equal(p, -1)
                r_mean = np.ma.mean(r_masked, axis=0)
                p_mean = np.ma.mean(p_masked, axis=0)
                average_recalls[key].append(r_mean)
                average_precisions[key].append(p_mean)
        
        final_recall = []
        final_precision = []
        for key in average_recalls.keys():
            r_mean = np.mean(average_recalls[key], axis=0)
            p_mean = np.mean(average_precisions[key], axis=0)
            final_recall.append(r_mean)
            final_precision.append(p_mean)
        
        final_recall = np.mean(np.array(final_recall), axis=0)
        final_precision = np.mean(np.array(final_precision), axis=0)
        p_mean = final_precision[::-1]
        r_mean = final_recall[::-1] 
        f = interp1d(r_mean, p_mean, assume_sorted=True, bounds_error=False, fill_value=(p_mean[0], p_mean[-1]))
        x = np.linspace(0, 1, 100)[::-1]
        y = f(x)
        ax.plot(x, y, label=weights, color=COLORS[weights])
        fig.savefig(f"./results/{args.dataset}-{args.mode}-{args.metric}.pdf", bbox_inches="tight")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.legend()
    fig.savefig(f"./results/{args.dataset}-{args.mode}-{args.metric}.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    main()

    

    