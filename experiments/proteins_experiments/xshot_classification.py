import numpy as np
import matplotlib.pyplot as plt 
import torch 

import matplotlib.pyplot as plt 
import argparse
from models.classifier import ClassificationHead
from utils.data_utils import load_theresa_proteins, fewshot_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.training_utils import AverageMeter, SaveBestModel
from models.intensity_model import get_intensity_model
import torchvision
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--class-type", "-ct", type=str, default="protein")
parser.add_argument("--pretraining", type=str, default="STED")
args = parser.parse_args()

def load_model(pretraining: str = "STED"):
    if pretraining == "STED":
        backbone = torchvision.models.resnet18()
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=1, bias=False)
        backbone.fc = torch.nn.Identity()
        checkpoint = torch.load("/home/frederic/Datasets/FLCDataset/baselines/result.pt")
        print(checkpoint.keys())
        model_dict = checkpoint["model"]["backbone"]
        backbone.load_state_dict(model_dict)
        model = ClassificationHead(
                backbone=backbone,
                output_channels=4
            )
    elif pretraining == "ImageNet":
        print("Loading ResNet-18 pre-trained on ImageNet")
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = torch.nn.Identity()
        model = ClassificationHead(
            backbone=backbone,
            output_channels=4,
        )
    elif pretraining == "Intensity":
        print("Loading IntensityModel")
        backbone = get_intensity_model()
    else:
        exit(f"Model {args.pretraining} not implemented yet.")
    return model

def compute_Nary_accuracy(preds: torch.Tensor, labels: torch.Tensor, N: int = 4) -> list:
    # accuracies = []
    correct = []
    big_n = []
    _, preds = torch.max(preds, 1)
    assert preds.shape == labels.shape
    c = torch.sum(preds == labels)
    correct.append(c.item())
    big_n.append(preds.shape[0])
    for n in range(N):
        c = ((preds == labels ) * (labels == n)).float().sum().cpu().detach().numpy()
        n = (labels==n).float().sum().cpu().detach().numpy()
        correct.append(c)
        big_n.append(n)
        # temp = ( (preds == labels) * (labels == n)).float().sum() / (labels == n).float().sum()
        # accuracies.append(temp.cpu().detach().numpy())
    return np.array(correct), np.array(big_n)

def validation_step(model, valid_loader, criterion, epoch, device):
    model.eval()
    loss_meter = AverageMeter()
    big_correct = np.array([0] * (4+1))
    big_n = np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, proteins, conditions in tqdm(valid_loader, desc="Validation..."):
            labels = proteins if args.class_type == "protein" else conditions
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


def train_one_epoch(train_loader, model, optimizer, criterion, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    for imgs, proteins, conditions in tqdm(train_loader, desc="Training..."):
        labels = proteins if args.class_type == "protein" else conditions
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

def plot_training_curves(train_loss, val_loss, val_acc, learning_rates, model_path):
    fig, axs = plt.subplots(2, 1, sharex=True)
    x = np.arange(0, len(train_loss), 1)
    ax1 = axs[0].twinx()
    ax2 = axs[0].twinx()
    axs[0].plot(x, train_loss, color='lightblue', label="Train")
    axs[0].plot(x, val_loss, color='lightcoral', label="Validation")
    ax1.plot(x, learning_rates, color='lightgreen', label='lr', ls='--')
    axs[1].plot(x, val_acc, color='lightcoral', label="Validation")
    axs[1].set_xlabel('Epochs')
    ax2.plot(x, learning_rates, color='palegreen', label='lr', ls='--', alpha=0.1)
    axs[1].set_ylabel('Accuracy')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].legend()
    fig.savefig(f"{model_path}/training_curves.png")
    plt.close(fig)


def train(model, train_loader, valid_loader, device, num_epochs, criterion, optimizer, scheduler, model_path):
    train_loss, val_loss, val_acc, lrates = [], [], [], []
    save_best_model = SaveBestModel(
        best=np.Inf,
        maximize=False,
        model_name="best_model",
        save_dir=model_path
    )
    for epoch in tqdm(range(num_epochs), desc="Epochs..."):
        loss = train_one_epoch(train_loader, model, optimizer, criterion, device, epoch)
        train_loss.append(loss)
        v_loss, v_acc = validation_step(model, valid_loader, criterion, epoch, device)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        scheduler.step(v_loss)
        temp_lr = optimizer.param_groups[0]['lr']
        lrates.append(temp_lr)
        save_best_model(v_loss, epoch=epoch, model=model, optimizer=optimizer, criterion=criterion)
        plot_training_curves(train_loss, val_loss, val_acc, lrates, model_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = fewshot_loader(
        path="/home/frederic/Datasets/FLCDataset",
        class_type=args.class_type,
        n_channels=1 if args.pretraining == "STED" else 3
    )
    model = load_model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5,
        # weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10, factor=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=500,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path="/home/frederic/Datasets/FLCDataset/baselines/proteins-finetuned"
    )


if __name__=="__main__":
    main()