import numpy as np 
import torch 
from torch import nn
from QualityNet.networks import NetTrueFCN 
import argparse 
from attribute_datasets import get_dataset  
from torch.utils.data import DataLoader 
from utils import AverageMeter, SaveBestModel
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser() 
parser.add_argument("--epochs", type=int, default=350)
args = parser.parse_args()

def validation_step(model, loader, device):
    model.eval()
    with torch.no_grad():
        loss_meter = AverageMeter()
        for images, metadata in loader:
            masks = metadata["mask"].to(device)
            images = images.to(device)
            targets = metadata["score"].to(device).float()
            _, outputs = model(images, masks)
            loss = criterion(outputs, targets)
            loss_meter.update(loss.item())
    return loss_meter.avg

def loss_tracker(train_loss: list, val_loss: list):
    fig = plt.figure()
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig("./QualityNet/trained_models/loss_tracker.png", bbox_inches="tight")
    plt.close(fig)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, valid_dataset = get_dataset(name="quality", training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=True, drop_last=True)

    model = NetTrueFCN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()

    train_loss, val_loss = [], []
    save_best_model = SaveBestModel(save_dir="./QualityNet/trained_models", model_name="qualitynet", maximize=False)
    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        for images, metadata in tqdm(train_dataloader):
            masks = metadata["mask"].to(DEVICE)
            images = images.to(DEVICE)
            targets = metadata["score"].to(DEVICE).float()
            optimizer.zero_grad()
            _, outputs = model(images, masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        valid_loss = validation_step(model=model, loader=valid_dataloader, device=DEVICE)
        train_loss.append(loss_meter.avg)
        val_loss.append(valid_loss)
        save_best_model(current_val=valid_loss, epoch=epoch, model=model, optimizer=optimizer, criterion=criterion)
        scheduler.step()
        print(f"Epoch {epoch + 1} - Train Loss: {loss_meter.avg} - Valid Loss: {valid_loss}")
        loss_tracker(train_loss=train_loss, val_loss=val_loss)


if __name__ == "__main__":
    main()

