import numpy as np 
import matplotlib.pyplot as plt 
import torch
import argparse 
from tqdm import tqdm 
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.insert(0, "../")
from loader import get_dataset
from model_builder import get_pretrained_model
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='synaptic-proteins')
parser.add_argument("--model", type=str, default="MAE")
parser.add_argument("--weights", type=str, default="STED")
parser.add_argument("--global-pool", type=str, default='avg')
args = parser.parse_args()

SAVE_EXPR = "linear-probe" if args.freeze else "finetuned"

def evaluate(model, loader, device):
    model.eval()
    big_correct = np.array([0] * (4+1))
    big_n = np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, proteins, conditions in tqdm(loader, desc="Evaluation..."):
            labels = proteins if args.class_type == 'protein' else conditions
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions = model(imgs)
            correct, n = compute_Nary_accuracy(predictions, labels)
            big_correct = big_correct + correct
            big_n = big_n + n
        accuracies = big_correct / big_n
        print("********* Validation metrics **********")
        print("Overall accuracy = {:.3f}".format(accuracies[0]))
        for i in range(1, 4+1):
            acc = accuracies[i]
            print("Class {} accuracy = {:.3f}".format(
                i, acc))
    return accuracies[0]

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
    for imgs, data_dict in tqdm(train_loader, desc="Training...."):
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
        test_loader,
        device,
        num_epochs, 
        criterion, 
        optimizer, 
        scheduler, 
        model_path
): 
    train_loss, val_loss, val_acc, lrates = [], [], [], []
    acc_per_epoch = []
    save_best_model = SaveBestModel(
        save_dir=model_path,
        model_name=f"{SAVE_EXPR}_model",
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
        if epoch == 1 or (epoch + 1) % 10 == 0:
            test_acc = evaluate(model=model, loader=test_loader, device=device)
            acc_per_epoch.append(test_acc)
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                }, f'{model_path}/{SAVE_EXPR}_epoch{epoch+1}_model.pth')
        temp_lr = optimizer.param_groups[0]['lr']
        lrates.append(temp_lr)
        save_best_model(v_loss, epoch=epoch, model=model, optimizer=optimizer, criterion=criterion)
        track_loss(train_loss, val_loss, val_acc, lrates, save_dir=f"{model_path}/{SAVE_EXPR}_curves.png")
    return acc_per_epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if args.weights == "ImageNet" else 1 
    train_loader, valid_loader, test_loader = get_dataset(name=args.dataset, transform=None, path=None, n_channels=n_channels, training=True)
    model = get_pretrained_model(
        name=args.model, 
        weights=args.weights, 
        path=None,
        freeze=args.freeze
        ).to(device)
    if args.freeze:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=20)
    criterion = torch.nn.CrossEntropyLoss()
    test_accuracies = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=200,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/finetuning/{args.pretraining}"
    )
    np.savez(
        f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/results/{SAVE_EXPR}/{args.pretraining}_test_results", 
        acc=test_accuracies)
    



if __name__=="__main__":
    main()