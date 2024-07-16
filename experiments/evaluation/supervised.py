import numpy as np 
import matplotlib.pyplot as plt
import torch 
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau 
from tqdm import tqdm
import argparse 
import sys 
sys.path.insert(0, "../")
from loaders import get_dataset 
from models import get_model
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss 
plt.style.use("dark_background")

parser = argparse.ArgumentParser() 
parser.add_argument("--dataset", type=str, default="neural-activity-states")
parser.add_argument("--model", type=str, default='mae-lightning-small')
parser.add_argument("--num-per-class", type=int, default=None)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

def set_seeds():
    np.random.seed(42)
    torch.manual_seed(42)

def validation_step(model, valid_loader, criterion, epoch, device):
    model.eval()
    loss_meter = AverageMeter()
    correct, N = np.array([0] * (4+1)), np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, data_dict in tqdm(valid_loader, desc="Validation..."):
            labels = data_dict["label"]
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
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")

    model, cfg = get_model(
        name=args.model,  
        mask_ratio=0.0,
        pretrained=False,
        in_channels=1,
        )
    
    batch_size = cfg.batch_size 

    train_loader, valid_loader, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=1, 
        batch_size=batch_size,
        training=True,
        num_samples=args.num_per_class
    )

    if "mae" in args.model.lower():
        model = model.backbone.vit

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.99))

    scheduler = ReduceLROnPlateau(optimizer, patience=5)
    criterion = torch.nn.CrossEntropyLoss()
    modelname = args.model.replace("-lightning", "")
    model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/supervised/{modelname}/{args.dataset}"

    train_loss, val_loss, val_acc, lrates = [], [], [], []
    save_best_model = SaveBestModel(
        save_dir=model_path,
        model_name="supervised"
    )

    for epoch in tqdm(range(1000), desc="Epochs..."):
        loss_meter = AverageMeter() 
        for imgs, data_dict in tqdm(train_loader, desc="Training..."):
            model.train()
            labels = data_dict["label"]
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad() 
            predictions = model(imgs)
            _, preds = torch.max(predictions, 1)
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

        if not args.dry_run:
            save_best_model(
                v_loss,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
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
            save_dir=f"{model_path}/supervised_training_curves.png"
        )


if __name__=="__main__":
    main()