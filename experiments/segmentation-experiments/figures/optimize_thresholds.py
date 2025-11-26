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

from stedfm.datasets import get_segmentation_dataset as get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--metric", type=str, default="f1")
parser.add_argument("--mode", type=str, default="pretrained-frozen")
parser.add_argument("--dataset", type=str, default="factin")
args = parser.parse_args()

class SegmentationConfiguration(Configuration):
    
    freeze_backbone: bool = True
    num_epochs: int = 300
    learning_rate: float = 1e-4

def num_classes():
    if args.dataset == "synaptic-semantic-segmentation":
        return 4 
    else:
        return 2

def compute_dice(pred_ch, y_ch, threshold):
    pred_binary = pred_ch > threshold 
    if not np.any(y_ch) and not np.any(pred_binary):
        return -1
    if np.unique(y_ch).size == 1:
        return -1
    intersection = np.logical_and(y_ch, pred_binary).sum()
    dice_score = (2 * intersection + 1) / (np.sum(y_ch) + np.sum(pred_binary) + 1)
    return dice_score

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
                region_scores.append(-1)
                continue
            if np.unique(pred_binary_crop).size == 1 and np.unique(target_crop).size > 1:
                region_scores.append(0.0)
                continue

            intersection = np.logical_and(target_crop, pred_binary_crop).sum()
            dice_score = (2 * intersection + 1) / (np.sum(target_crop) + np.sum(pred_binary_crop) + 1)
            region_scores.append(dice_score)
        region_scores = np.array(region_scores)
        region_scores = np.ma.masked_equal(region_scores, -1)
        mean_region_score = np.ma.mean(region_scores)
        return mean_region_score

def compute_iou(pred_ch, y_ch, threshold):
    pred_binary = pred_ch > threshold
    if not np.any(y_ch) and not np.any(pred_binary):
        return -1
    if np.unique(y_ch).size == 1:
        return -1
    intersection = np.logical_and(y_ch, pred_binary).sum()
    union = np.logical_or(y_ch, pred_binary).sum()
    iou_score = intersection / (union + 1e-8)
    return iou_score

def compute_weak_iou(pred_ch, target_ch, threshold):
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
                region_scores.append(-1)
                continue
            if np.unique(pred_binary_crop).size == 1 and np.unique(target_crop).size > 1:
                region_scores.append(0.0)
                continue
            intersection = np.logical_and(target_crop, pred_binary_crop).sum()
            union = np.logical_or(target_crop, pred_binary_crop).sum()
            iou_score = intersection / (union + 1e-8)
            region_scores.append(iou_score)
        region_scores = np.array(region_scores)
        region_scores = np.ma.masked_equal(region_scores, -1)
        mean_region_score = np.ma.mean(region_scores)
        return mean_region_score

def compute_recall_precision(pred_ch, y_ch, threshold):  
    pred_binary = pred_ch > threshold
    if not np.any(y_ch) and not np.any(pred_binary):
        return -1, -1
    if np.unique(y_ch).size == 1:
        return -1, -1
    tp = np.logical_and(pred_binary, y_ch).sum()
    fp = np.logical_and(pred_binary, np.logical_not(y_ch)).sum()
    fn = np.logical_and(np.logical_not(pred_binary), y_ch).sum()
    recall_score = tp / (tp + fn + 1e-8)
    precision_score = tp / (tp + fp + 1e-8)
    return recall_score, precision_score

def compute_weak_recall_precision(pred_ch, target_ch, threshold):
    pred_binary = pred_ch > threshold
    target_ch = target_ch > 0.5
    target_label = measure.label(target_ch)
    target_rprops = measure.regionprops(target_label)
    if len(target_rprops) == 0:
        return -1, -1
    else:
        region_p_scores, region_r_scores = [], []
        for region in target_rprops:
            ymin, xmin, ymax, xmax = region.bbox
            ymin = max(0, ymin - 10)
            ymax = min(target_ch.shape[0], ymax + 10)
            xmin = max(0, xmin - 10)
            xmax = min(target_ch.shape[1], xmax + 10)
            target_crop = target_ch[ymin:ymax, xmin:xmax].ravel()
            pred_binary_crop = pred_binary[ymin:ymax, xmin:xmax].ravel()
            if np.unique(target_crop).size == 1:
                region_r_scores.append(-1)
                region_p_scores.append(-1)
                continue
            if np.unique(pred_binary_crop).size == 1 and np.unique(target_crop).size > 1:
                region_r_scores.append(0.0)
                region_p_scores.append(0.0)
                continue
            tp = np.logical_and(pred_binary_crop, target_crop).sum()
            fp = np.logical_and(pred_binary_crop, np.logical_not(target_crop)).sum()
            fn = np.logical_and(np.logical_not(pred_binary_crop), target_crop).sum()
            recall_score = tp / (tp + fn + 1e-8)
            precision_score = tp / (tp + fp + 1e-8)
            region_r_scores.append(recall_score)
            region_p_scores.append(precision_score)
        region_r_scores = np.array(region_r_scores)
        region_p_scores = np.array(region_p_scores)
        region_r_scores = np.ma.masked_equal(region_r_scores, -1)
        region_p_scores = np.ma.masked_equal(region_p_scores, -1)
        mean_region_r_score = np.ma.mean(region_r_scores)
        mean_region_p_score = np.ma.mean(region_p_scores)
        return mean_region_r_score, mean_region_p_score

def compute_f1(pred_ch, y_ch, threshold):
    pred_binary = pred_ch > threshold
    if not np.any(y_ch) and not np.any(pred_binary):
        return -1
    if np.unique(y_ch).size == 1:
        return -1
    tp = np.logical_and(pred_binary, y_ch).sum()
    fp = np.logical_and(pred_binary, np.logical_not(y_ch)).sum()
    fn = np.logical_and(np.logical_not(pred_binary), y_ch).sum()
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1_score

def compute_weak_f1(pred_ch, target_ch, threshold):
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
                region_scores.append(-1)
                continue
            if np.unique(pred_binary_crop).size == 1 and np.unique(target_crop).size > 1:
                region_scores.append(0.0)
                continue
            tp = np.logical_and(pred_binary_crop, target_crop).sum()
            fp = np.logical_and(pred_binary_crop, np.logical_not(target_crop)).sum()
            fn = np.logical_and(np.logical_not(pred_binary_crop), target_crop).sum()
            f1_score = 2 * tp / (2 * tp + fp + fn + 1e-8)
            region_scores.append(f1_score)
        region_scores = np.array(region_scores)
        region_scores = np.ma.masked_equal(region_scores, -1)
        mean_region_score = np.ma.mean(region_scores)
        return mean_region_score

def compute_f1_curve(pred, y, thresholds):
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.squeeze().cpu().numpy() 

    f1_per_class = {key: [] for key in range(pred.shape[0])}
    for threshold in thresholds:
        for ch in range(pred.shape[0]):
            pred_ch, y_ch = pred[ch], y[ch]
            f1_score = compute_f1(pred_ch, y_ch, threshold)
            f1_per_class[ch].append(f1_score)
    return f1_per_class
            
def compute_weak_f1_curve(pred, target, thresholds):
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.squeeze().cpu().numpy()

    f1_per_class = {key: [] for key in range(pred.shape[0])}
    for threshold in thresholds:
        for ch in range(pred.shape[0]):
            pred_ch, target_ch = pred[ch], target[ch]
            mean_region_score = compute_weak_f1(pred_ch, target_ch, threshold)
            f1_per_class[ch].append(mean_region_score)
    return f1_per_class
    

def main():
    os.makedirs(f"./results_v2", exist_ok=True)
    THRESHOLDS = np.linspace(0.01, 0.99, 100) 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_weights = [None, "MAE_SMALL_IMAGENET1K_V1", "MAE_SMALL_JUMP", "MAE_SMALL_HPA", "MAE_SMALL_SIM", "MAE_SMALL_STED"]

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
            drop_last=False,
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

        print("========================================")
        print(f"Optimizing thresholds for weights: {weights}")
        print(f"Mode: {args.mode}")
        print(f"Dataset: {args.dataset}")
        print(f"Backbone weights: {weights}")
        print(f"Length of validation set: {len(valid_dataset)}")
        print(f"Length of test set: {len(test_dataset)}")
        print("========================================")

        if weights is None:
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

        if weights is None:
            weights = "from-scratch"


        for path in model_paths:
            f1_scores = defaultdict(list)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                for X, y in tqdm(valid_loader, desc=f"... Optimizing {weights} ...", total=len(valid_loader)):
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X)
                    f1_per_class = compute_weak_f1_curve(pred, y, THRESHOLDS) if args.dataset == "synaptic-semantic-segmentation" else compute_f1_curve(pred, y, THRESHOLDS)
                    for key in f1_per_class.keys():
                        f1_scores[key].append(f1_per_class[key])

            for key in f1_scores.keys():
                s = np.array(f1_scores[key])
                x = THRESHOLDS
                s_masked = np.ma.masked_equal(s, -1)
                mean = np.ma.mean(s_masked, axis=0)
                average_scores[key].append(mean)

        temp_thresholds = []
        for key in average_scores.keys():
            mean = np.mean(average_scores[key], axis=0) 
            max_f1 = np.max(mean)
            max_threshold = THRESHOLDS[np.argmax(mean)]
            temp_thresholds.append(max_threshold)

        print("Optimized thresholds:")
        print(temp_thresholds)

        ### Compute all metrics with thresholds optimized against F1 score on the validation set
        weight_f1 = defaultdict(list)
        weight_dice = defaultdict(list)
        weight_iou = defaultdict(list)
        weight_precision = defaultdict(list)
        weight_recall = defaultdict(list)
        for p, path in enumerate(model_paths):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                for X, y in tqdm(test_loader, desc=f"... Evaluating {weights} ...", total=len(test_loader)):
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X)
                    pred_numpy = pred.squeeze().cpu().numpy() 
                    y_numpy = y.squeeze().cpu().numpy()

                    f1_ch_scores = []
                    dice_ch_scores = []
                    iou_ch_scores = []
                    precision_ch_scores = []
                    recall_ch_scores = []
                    for ch in range(pred_numpy.shape[0]):
                        best_weight_threshold = temp_thresholds[ch]
                        pred_ch = pred_numpy[ch]
                        y_ch = y_numpy[ch]
                        # F1
                        temp_f1 = compute_weak_f1(pred_ch, y_ch, best_weight_threshold) if args.dataset == "synaptic-semantic-segmentation" else compute_f1(pred_ch, y_ch, best_weight_threshold)
                        if temp_f1 != -1:
                            f1_ch_scores.append(temp_f1)
                        
                        # Dice 
                        temp_dice = compute_weak_dice(pred_ch, y_ch, best_weight_threshold) if args.dataset == "synaptic-semantic-segmentation" else compute_dice(pred_ch, y_ch, best_weight_threshold)
                        if temp_dice != -1:
                            dice_ch_scores.append(temp_dice)

                        # iou 
                        temp_iou = compute_weak_iou(pred_ch, y_ch, best_weight_threshold) if args.dataset == "synaptic-semantic-segmentation" else compute_iou(pred_ch, y_ch, best_weight_threshold)
                        if temp_iou != -1:
                            iou_ch_scores.append(temp_iou)

                        # Recall and Precision 
                        temp_recall, temp_precision = compute_weak_recall_precision(pred_ch, y_ch, best_weight_threshold) if args.dataset == "synaptic-semantic-segmentation" else compute_recall_precision(pred_ch, y_ch, best_weight_threshold)
                        if temp_recall != -1:
                            recall_ch_scores.append(temp_recall)
                        if temp_precision != -1:
                            precision_ch_scores.append(temp_precision)

                    if len(f1_ch_scores) == 0:
                        continue
                    f1_score = np.mean(f1_ch_scores)
                    weight_f1[p].append(f1_score) 

                    dice_score = np.mean(dice_ch_scores)
                    weight_dice[p].append(dice_score)

                    iou_score = np.mean(iou_ch_scores)
                    weight_iou[p].append(iou_score)

                    recall_score = np.mean(recall_ch_scores)
                    weight_recall[p].append(recall_score)
                    
                    precision_score = np.mean(precision_ch_scores)
                    weight_precision[p].append(precision_score)

        f1_sum = []
        dice_sum = []
        iou_sum = []
        recall_sum = []
        precision_sum = []
        for key in weight_f1.keys():
            f1_sum.append(np.mean(weight_f1[key], axis=0))
            dice_sum.append(np.mean(weight_dice[key], axis=0))
            iou_sum.append(np.mean(weight_iou[key], axis=0))
            recall_sum.append(np.mean(weight_recall[key], axis=0))
            precision_sum.append(np.mean(weight_precision[key], axis=0))

        final_f1 = np.mean(np.array(f1_sum), axis=0)
        final_dice = np.mean(np.array(dice_sum), axis=0)
        final_iou = np.mean(np.array(iou_sum), axis=0)
        final_recall = np.mean(np.array(recall_sum), axis=0)
        final_precision = np.mean(np.array(precision_sum), axis=0)
        temp_thresholds = [round(threshold, 4) for threshold in temp_thresholds]
        with open(f"./results_v2/{args.dataset}-{args.mode}-{args.metric}.txt", "a") as f:
            f.write(f"----------------{weights}----------------\n")
            f.write(f"\tF1 score: {final_f1:.4f}\n")
            f.write(f"\tDice score: {final_dice:.4f}\n")
            f.write(f"\tIoU score: {final_iou:.4f}\n")
            f.write(f"\tRecall score: {final_recall:.4f}\n")
            f.write(f"\tPrecision score: {final_precision:.4f}\n")
            f.write("----------------------------------------\n")

                        
if __name__ == "__main__":
    main()