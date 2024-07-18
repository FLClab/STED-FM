import torch 
import matplotlib.pyplot as plt 
import numpy as np
import argparse 
from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim 
from torch.utils.data import DataLoader 
from decoders.vit import ViTDecoder
from datasets import get_dataset 
import os 
import sys 
sys.path.insert(0, "../")
from model_builder import get_pretrained_model_v2
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss, compute_mean_average_precision
plt.style.use("dark_background")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="zooniverse")
parser.add_argument("--backbone", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--global-pool", type=str, default="avg")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

def set_seeds():
    np.random.seed(42)
    torch.manual_seed(42)

def get_save_folder() -> str: 
    if "imagenet" in args.weights.lower():
        return "ImageNet"
    elif "sted" in args.weights.lower():
        return "STED"
    elif "jump" in args.weights.lower():
        return "JUMP"
    elif "ctc" in args.weights.lower():
        return "CTC"
    elif "hpa" in args.weights.lower():
        return "HPA"
    else:
        raise NotImplementedError("The requested weights do not exist.")
    

def display_predictions(imgs: torch.Tensor, masks: torch.Tensor, predictions: torch.Tensor):
    SAVE_NAME = get_save_folder()
    counter = 0
    for img, mask, pred in zip(imgs, masks, predictions):
        if counter > 20:
            break
        else:

            img = img.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            print(img.shape, mask.shape, pred.shape)
            fig, axs = plt.subplot_mosaic([
                ["img", "img", "img", "img", "img", "img"],
                ["gt_round", "gt_elongated", "gt_multidomain", "gt_irregular", "gt_perforated", "gt_noise"],
                ["p_round", "p_elongated", "p_multidomain", "p_irregular", "p_perforated", "p_noise"]
            ], figsize=(20, 10))
            axs["img"].imshow(img[0], cmap='hot', vmax=1.0)
            axs["gt_round"].imshow(mask[0], cmap='gray')
            axs["gt_elongated"].imshow(mask[1], cmap='gray')
            axs["gt_multidomain"].imshow(mask[2], cmap='gray')
            axs["gt_irregular"].imshow(mask[3], cmap='gray')
            axs["gt_perforated"].imshow(mask[4], cmap='gray')
            axs["gt_noise"].imshow(mask[5], cmap='gray')
            axs["p_round"].imshow(pred[0], cmap='gray')
            axs["p_elongated"].imshow(pred[1], cmap='gray')
            axs["p_multidomain"].imshow(pred[2], cmap='gray')
            axs["p_irregular"].imshow(pred[3], cmap='gray')
            axs["p_perforated"].imshow(pred[4], cmap='gray')
            axs["p_noise"].imshow(pred[5], cmap='gray')
           
            fig.savefig(f"./results/{args.backbone.replace('-lightning', '')}_{SAVE_NAME}/temp_{counter}.png", dpi=1200, bbox_inches='tight')
            plt.close(fig)
            counter += 1
    
def validation_step(model, valid_loader, criterion, epoch, device):
    model.eval()
    loss_meter = AverageMeter()
    average_precision = np.zeros((6,))
    counts = np.zeros((6,))
    with torch.no_grad():
        for imgs, masks in tqdm(valid_loader, desc="...Validation..."):
            imgs, labels = imgs.to(device), masks.to(device)
            predictions = model(imgs)
            mAP = compute_mean_average_precision(preds=predictions, labels=masks)
            average_precision = average_precision + mAP
            counts = counts + np.ones((6,))
            loss = criterion(predictions, labels)
            loss_meter.update(loss.item())
    mean_average_precision = average_precision / counts
    mmap = np.mean(mean_average_precision)
    print("********* Validation metrics **********")
    print("Epoch {} validation loss = {:.3f} ({:.3f})".format(
        epoch + 1, loss_meter.val, loss_meter.avg))
    print("Overall mAP = {:.3f}".format(mmap))
    classes = ["Round", "Elongated", "Multidomain", "Irregular", "Perforated", "Noise"]
    for i in range(mean_average_precision.shape[0]):
        acc = mean_average_precision[i]
        print(f"\t{classes[i]} mAP = {acc}")
    return loss_meter.avg, mmap

def main():
    SAVENAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3 if "imagenet" in args.weights.lower() else 1
    backbone, cfg = get_pretrained_model_v2(
        name=args.backbone, 
        weights=args.weights,
        as_classifier=False,
        blocks='all',
        path=None,
        mask_ratio=0.0, 
        pretrained=True if "imagenet" in args.weights.lower() else False,
        in_channels=n_channels,
        )
    backbone = backbone.backbone
    cfg.batch_size = 32
    
    print(f"--- Loaded backbone ---")
    train_dataset, val_dataset, _ = get_dataset(
        name=args.dataset, cfg=cfg, n_channels=n_channels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
    )

    cfg.backbone = "mae"
    model = ViTDecoder(backbone=backbone, cfg=cfg, in_channels=n_channels, out_channels=6, extract_layers=[3, 6, 9, 12])
    model = model.to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=10, threshold=0.01, min_lr=1e-5, factor=0.9)

    modelname = args.backbone.replace("-lightning", "")
    model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/segmentation-baselines/{modelname}_{SAVENAME}/{args.dataset}"

    train_loss, val_loss, val_acc, lrates = [], [], [], []
    save_best_model = SaveBestModel(
        save_dir=model_path,
        model_name="segmentor"
    )


    for epoch in tqdm(range(100), desc="Epochs..."):
        loss_meter = AverageMeter()
        for imgs, masks in tqdm(train_loader, desc="Training..."):
            model.train()
            imgs, labels = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            predictions = model(imgs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        if (epoch + 1) % 20 == 0:
            display_predictions(imgs=imgs, masks=labels, predictions=predictions)
            

        v_loss, v_acc = validation_step(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            epoch=epoch,
            device=device
        )

        # Do not save best model if in a dry run
        if not args.dry_run:
            save_best_model(
                v_loss,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                criterion=criterion
                )
        temp_lr = optimizer.param_groups[0]['lr']
        lrates.append(temp_lr)
        train_loss.append(loss_meter.avg)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        scheduler.step(v_loss)
        track_loss(
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            lrates=lrates,
            save_dir=f"{model_path}/segmentor_training-curves.png"
        )



if __name__=="__main__":
    main()