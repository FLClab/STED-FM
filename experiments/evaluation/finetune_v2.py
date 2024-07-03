import numpy as np
import matplotlib.pyplot as plt
import torch 
from tqdm import tqdm 
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse 
import sys 
from torchinfo import summary
sys.path.insert(0, "../")
from loaders import get_dataset
from model_builder import get_pretrained_model_v2 
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss

plt.style.use("dark_background")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='synaptic-proteins')
parser.add_argument("--model", type=str, default='mae-lightning-small')
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--global-pool", type=str, default='avg')
parser.add_argument("--blocks", type=str, default="all") # linear-probing by default
parser.add_argument("--track-epochs", action="store_true")
parser.add_argument("--num-per-class", type=int, default=10)
args = parser.parse_args()

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
    
def validation_step(model, valid_loader, criterion, epoch, device):
    model.eval()
    loss_meter = AverageMeter()
    correct, N = np.array([0] * (4+1)), np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, data_dict in tqdm(valid_loader, desc="Validation..."):
            labels =  data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions = model(imgs)
            loss = criterion(predictions, labels)
            loss_meter.update(loss.item())
            c, n = compute_Nary_accuracy(predictions, labels)
            correct = correct + c 
            N = n + N
    accuracies = correct / N 
    print("********* Validation metrics **********")
    print("Epoch {} validation loss = {:.3f} ({:.3f})".format(
        epoch + 1, loss_meter.val, loss_meter.avg))
    print("Overall accuracy = {:.3f}".format(accuracies[0]))
    for i in range(1, 4+1):
        acc = accuracies[i]
        print("Class {} accuracy = {:.3f}".format(
            i, acc))
    return loss_meter.avg, accuracies[0]


def main():
    SAVENAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if SAVENAME == "ImageNet" else 1

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if n_channels==3 else 1,
        in_channels=n_channels,
        as_classifier=True,
        blocks=args.blocks
    )
    batch_size = cfg.batch_size

    train_loader, valid_loader, _ = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        batch_size=batch_size,
        training=True,
        num_samples=args.num_per_class
    )

    num_epochs = 100
    model = model.to(device)
    if args.blocks == "all":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0, momentum=0.9) 
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
    criterion = torch.nn.CrossEntropyLoss()
    modelname = args.model.replace("-lightning", "")
    model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{modelname}_{SAVENAME}/{args.dataset}"

    # Training loop
    train_loss, val_loss, val_acc, lrates = [], [], [], []
    save_best_model = SaveBestModel(
        save_dir=model_path,
        model_name=f"linear-probe"
    )
    for epoch in tqdm(range(num_epochs), desc="Epochs..."):
        loss_meter = AverageMeter()
        for imgs, data_dict in tqdm(train_loader, desc="Training..."):
            model.train()
            labels = data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            predictions = model(imgs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
        v_loss, v_acc = validation_step(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            epoch=epoch,
            device=device
        )
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
        scheduler.step()
        track_loss(
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            lrates=lrates,
            save_dir=f"{model_path}/linear-probe_training-curves.png"
        )
            
if __name__=="__main__":
    main()