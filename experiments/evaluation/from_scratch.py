import torch 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sys 
sys.path.insert(0, "../")
from loaders import get_dataset
from model_builder import get_pretrained_model, get_base_model
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="synaptic-proteins")
parser.add_argument("--model", type=str, default="vit-small")
args = parser.parse_args()

def validation_step(
        model,
        valid_loader,
        criterion,
        epoch,
        device
):
    model.eval()
    loss_meter = AverageMeter()
    big_correct = np.array([0] * (4+1))
    big_n = np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, data_dict in tqdm(valid_loader, desc="Validation..."):
            labels = data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions = model(imgs)
            loss = criterion(predictions, labels)
            loss_meter.update(loss.item())
            correct, n = compute_Nary_accuracy(predictions, labels)
            big_correct = big_correct + correct
            big_n = big_n + n
        accuracies = big_correct / big_n
        print("********* Validation metrics **********")
        print("Epoch {} validation loss = {:.3f} ({:.3f})".format(
            epoch + 1, loss_meter.val, loss_meter.avg))
        print("Overall accuracy = {:.3f}".format(accuracies[0]))
        for i in range(1, 4+1):
            acc = accuracies[i]
            print("Class {} accuracy = {:.3f}".format(
                i, acc))
    return loss_meter.avg, accuracies[0]

def train_one_epoch(
        train_loader,
        model,
        optimizer, 
        criterion,
        device,
        epoch
):
    model.train()
    loss_meter = AverageMeter()
    for imgs, data_dict in tqdm(train_loader, desc="Training..."):
        labels = data_dict['label']
        imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    print("Epoch {} loss = {:.3f} ({:.3f})".format(
        epoch + 1, loss_meter.val, loss_meter.avg))
    # print(f"The number of images sampled per class for this epoch was: {class_counts}")
    return loss_meter.avg

def train(
        model,
        train_loader,
        valid_loader,
        device, 
        num_epochs,
        criterion,
        optimizer,
        scheduler,
        model_path
): 
    train_loss, val_loss, val_acc, lrates = [], [], [], []
    save_best_model = SaveBestModel(
        save_dir=model_path,
        model_name=f'{args.model}_from-scratch_model'
    )
    for epoch in tqdm(range(num_epochs), desc="Epochs..."):
        loss = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            device=device
        )
        train_loss.append(loss)
        v_loss, v_acc = validation_step(model, valid_loader, criterion, epoch, device)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        scheduler.step()
        temp_lr = optimizer.param_groups[0]['lr']
        lrates.append(temp_lr)
        save_best_model(v_loss, epoch=epoch, model=model, optimizer=optimizer, criterion=criterion)
        track_loss(train_loss, val_loss, val_acc, lrates, save_dir=f"{model_path}/{args.model}_from-scratch_curves.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    train_loader, valid_loader, _ = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=1,
        training=True
    )
    model, cfg = get_base_model(name=args.model)
    model = model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.3, betas=(0.9, 0.95)) --> proposed by the MAE paper, but does not seem to be optimal here
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100)
    criterion = torch.nn.CrossEntropyLoss()
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=1000,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/MAE_fully-supervised/{args.dataset}"
    )

if __name__=="__main__":
    main()