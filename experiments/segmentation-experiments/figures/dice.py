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
from skimage import measure
import sys 
sys.path.insert(0, "../")
from datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--metric", type=str, default="dice")
parser.add_argument("--mode", type=str, default="pretrained-frozen")
parser.add_argument("--dataset", type=str, default="factin")
args = parser.parse_args()


class SegmentationConfiguration(Configuration):
    
    freeze_backbone: bool = True
    num_epochs: int = 300
    learning_rate: float = 1e-4


def compute_dice(pred_ch, target_ch, threshold):
    pred_binary = pred_ch > threshold 
    if not np.any(target_ch) and not np.any(pred_binary):
        return -1 
    if np.unique(target_ch).size == 1:
        return -1 
    intersection = np.logical_and(target_ch, pred_binary).sum()
    dice_score = (2 * intersection + 1) / (np.sum(target_ch) + np.sum(pred_binary) + 1)
    return dice_score


def compute_dice_curve(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.squeeze().cpu().numpy()
    thresholds = np.linspace(0.01, 0.99, 100)
    dice_per_class = {key: [] for key in range(pred.shape[0])}
    for threshold in thresholds:
        for ch in range(pred.shape[0]):
            pred_ch, target_ch = pred[ch], target[ch]
            dice_score = compute_dice(pred_ch, target_ch, threshold)
            dice_per_class[ch].append(dice_score)
    return dice_per_class


def compute_weak_dice(pred_ch, target_ch, threshold):
    pred_binary = pred_ch > threshold
    target_ch = target_ch > 0.5
    target_label = measure.label(target_ch)
    target_rprops = measure.regionprops(target_label)

    if len(target_rprops) == 0:
        return -1
    else:
        region_scores = []
        for region in target_rprops:
            ymin, xmin, ymax, xmax = region.bbox 
            ymin = max(0, ymin - 10)
            ymax = min(target_ch.shape[0], ymax + 10)
            xmin = max(0, xmin - 10)
            xmax = min(target_ch.shape[1], xmax + 10)
            target_crop = target_ch[ymin:ymax, xmin:xmax].ravel()
            pred_binary_crop = pred_binary[ymin:ymax, xmin:xmax].ravel()
            if np.unique(target_crop).size == 1:
                return -1
            if np.unique(pred_binary_crop).size == 1 and np.unique(target_crop).size > 1:
                return 0.0

            intersection = np.logical_and(target_crop, pred_binary_crop).sum()
            dice_score = (2 * intersection + 1) / (np.sum(target_crop) + np.sum(pred_binary_crop) + 1)
            region_scores.append(dice_score)
        region_scores = np.array(region_scores)
        region_scores = np.ma.masked_equal(region_scores, -1)
        mean_region_score = np.ma.mean(region_scores)
        return mean_region_score



def compute_weak_dice_curve(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.squeeze().cpu().numpy()

    thresholds = np.linspace(0.01, 0.99, 100)
    dice_per_class = {key: [] for key in range(pred.shape[0])}
    for threshold in thresholds:
        for ch in range(pred.shape[0]):
            pred_ch, target_ch = pred[ch], target[ch]
            mean_region_score = compute_weak_dice(pred_ch, target_ch, threshold)
            dice_per_class[ch].append(mean_region_score)
    return dice_per_class

def num_classes():
    if args.dataset == "synaptic-semantic-segmentation":
        return 4 
    else:
        return 2
                    
            
def main():

    THRESHOLDS = np.linspace(0.01, 0.99, 100)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    if args.mode == "from-scratch":
        backbone_weights = [None] 
    else:
        backbone_weights = ["MAE_SMALL_IMAGENET1K_V1", "MAE_SMALL_JUMP", "MAE_SMALL_HPA", "MAE_SMALL_SIM", "MAE_SMALL_STED"]

    fig, axs = plt.subplots(1, num_classes(), figsize=(10, 5))

    best_thresholds = {key: None for key in backbone_weights}
    for weights in backbone_weights:
        backbone, cfg = get_pretrained_model_v2(
            name=args.model,
            weights=weights,
        )
        _, valid_dataset, test_dataset = get_dataset(
            name=args.dataset, 
            cfg=cfg,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1, 
            shuffle=False,
            drop_last=False
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

        if args.mode == "from-scratch":

            model_paths = glob.glob(f"{BASE_PATH}/segmentation-baselines/{args.model}/{args.dataset}/from-scratch-*/result.pt", recursive=True)
            model_paths = [p for p in model_paths if "labels" not in p]
            print(f"Found {len(model_paths)} model paths for {args.mode} mode")
        else:
            model_paths = glob.glob(f"{BASE_PATH}/segmentation-baselines/{args.model}/{args.dataset}/{args.mode}-{weights}-*/result.pt", recursive=True)
            model_paths = [p for p in model_paths if "labels" not in p]
            if args.mode == "pretrained":
                model_paths = [p for p in model_paths if "frozen" not in p]

        model_paths = list(set(model_paths))
        average_scores = defaultdict(list)
        if args.mode == "from-scratch":
            weights = args.mode
        for path in model_paths:
            dice_scores = defaultdict(list)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                for X, y in tqdm(valid_loader, desc=f"Processing {weights}"):
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X)
                    dice_per_class = compute_weak_dice_curve(pred, y) if args.dataset == "synaptic-semantic-segmentation" else compute_dice_curve(pred, y)
                    for key in dice_per_class.keys():
                        dice_scores[key].append(dice_per_class[key])

            for key in dice_scores.keys():
                # print(dice_scores[key])
                s = np.array(dice_scores[key])
                x = np.linspace(0.01, 0.99, 100)
                s_masked = np.ma.masked_equal(s, -1)
                mean = np.ma.mean(s_masked, axis=0)
                axs[key].plot(x, mean, color=COLORS[weights], alpha=0.4)
                average_scores[key].append(mean)

        temp_thresholds = []
        for key in average_scores.keys():
            mean = np.mean(average_scores[key], axis=0) 
            max_dice = np.max(mean)
            max_threshold = THRESHOLDS[np.argmax(mean)]
            temp_thresholds.append(max_threshold)
            
        # best_weight_threshold = np.mean(temp_thresholds)
        weight_dice = defaultdict(list)
        for p, path in enumerate(model_paths):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                for X, y in tqdm(test_loader, desc="... Evaluating best threshold ...", total=len(test_loader)):
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X)
                    pred_numpy = pred.squeeze().cpu().numpy()
                    y_numpy = y.squeeze().cpu().numpy()
                    dice_ch_scores = []
                    for ch in range(pred_numpy.shape[0]):
                        best_weight_threshold = temp_thresholds[ch]
                        pred_ch = pred_numpy[ch]
                        y_ch = y_numpy[ch]
                        temp = compute_weak_dice(pred_ch, y_ch, best_weight_threshold) if args.dataset == "synaptic-semantic-segmentation" else compute_dice(pred_ch, y_ch, best_weight_threshold)
                        if temp != -1:
                            dice_ch_scores.append(temp)
                    dice_score = np.mean(dice_ch_scores)
                    weight_dice[p].append(dice_score)

        dice_sum = []
        for key in weight_dice.keys():
            dice_sum.append(np.mean(weight_dice[key], axis=0))
        final_dice = np.mean(np.array(dice_sum), axis=0)
        temp_thresholds = [round(threshold, 4) for threshold in temp_thresholds]
        with open(f"./results/{args.dataset}-{args.mode}-{args.metric}.txt", "a") as f:
            f.write(f"----------------{weights}----------------\n")
            f.write(f"\tAfter optimizing threshold on the validation set, dice score on the test set is: {final_dice:.4f}\n")
            f.write(f"\tBest thresholds were: {temp_thresholds}\n")
            f.write("----------------------------------------\n")
        print(f"----------------{weights}----------------")
        print(f"\tAfter optimizing threshold on the validation set, dice score on the test set is: {final_dice:.4f}")
        print(f"\tBest threshold were: {temp_thresholds}")
        print("----------------------------------------")
    
    for ax in axs:
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Dice score")
    fig.savefig(f"./results/{args.dataset}-{args.mode}-{args.metric}.pdf", bbox_inches="tight")








        
    # for ax in axs:
    #     ax.set_xlabel("Threshold")
    #     ax.set_ylabel("Dice Score")
    # fig.savefig(f"./results/{args.dataset}-{args.mode}-{args.metric}.png", bbox_inches="tight")
    # plt.close(fig)
        
                      


if __name__=="__main__":
    main()
