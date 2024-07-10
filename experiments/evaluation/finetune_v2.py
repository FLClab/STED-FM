import numpy as np
import matplotlib.pyplot as plt
import torch 
from tqdm import tqdm 
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import argparse 
import sys 
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
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
parser.add_argument("--num-per-class", type=int, default=None)
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

def knn_sanity_check(
        model: torch.nn.Module, 
        loader: torch.utils.data.DataLoader, 
        device: torch.device, 
        savename: str,
        epoch: int,
        ) -> None:
    samples, ground_truth = [], []
    model.eval()
    with torch.no_grad():
        for x, data_dict in tqdm(loader, desc="Extracting features..."):
            labels = data_dict['label']
            x, labels = x.to(device), labels.to(device) 
            if "mae" in args.model.lower():
                out, features = model(x)
                feat = features.data.cpu().numpy()
                truth = labels.data.cpu().numpy()
                

            truth = labels.data.cpu().numpy()
            feat = features.data.cpu().numpy()
            ground_truth.extend(truth)
            samples.extend(feat)

    samples = np.array(samples)
    ground_truth = np.array(ground_truth).astype(np.int64)


    neighbors_obj = NearestNeighbors(n_neighbors=6, metric="precomputed")
    pdistances = cdist(samples, samples)
    neighbors_obj = neighbors_obj.fit(pdistances)
    distances, neighbors = neighbors_obj.kneighbors(X=pdistances, return_distance=True)
    neighbors = neighbors[:, 1:]

    associated_labels = ground_truth[neighbors]
    uniques = np.unique(ground_truth).astype(np.int64)
    
    confusion_matrix = np.zeros((len(uniques), len(uniques)))
    for unique in tqdm(uniques, desc="Confusion matrix computation..."):
        mask = ground_truth == unique
        for predicted_unique in uniques:
            votes = np.sum((associated_labels[mask] == predicted_unique).astype(int), axis=-1)
            confusion_matrix[unique, predicted_unique] += np.sum(votes >= 3)
    accuracy = np.diag(confusion_matrix).sum() / np.sum(confusion_matrix)
    print(f"--- Epoch {epoch} --> {args.dataset} ; {args.model} ; {savename} ---\n\tAccuracy: {accuracy * 100:0.2f}\n")

    
def validation_step(model, valid_loader, criterion, epoch, device):
    model.eval()
    loss_meter = AverageMeter()
    correct, N = np.array([0] * (4+1)), np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, data_dict in tqdm(valid_loader, desc="Validation..."):
            labels =  data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions, _ = model(imgs)
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
    summary(model, input_size=(batch_size, n_channels, 224, 224))

    train_loader, valid_loader, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        batch_size=32,
        training=True,
        num_samples=args.num_per_class,
    )

    num_epochs = 100
    model = model.to(device)
    if args.blocks == "all":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0, momentum=0.9) 
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.99))
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=5)
    criterion = torch.nn.CrossEntropyLoss()
    modelname = args.model.replace("-lightning", "")
    model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{modelname}_{SAVENAME}/{args.dataset}"

    # Training loop
    train_loss, val_loss, val_acc, lrates = [], [], [], []
    save_best_model = SaveBestModel(
        save_dir=model_path,
        model_name="linear-probe" if args.blocks == "all" else "finetuned"
    )

    # knn_sanity_check(model=model, loader=test_loader, device=device, savename=SAVENAME, epoch=0)

    for epoch in tqdm(range(num_epochs), desc="Epochs..."):
        loss_meter = AverageMeter()
        for imgs, data_dict in tqdm(train_loader, desc="Training..."):
            model.train()
            labels = data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            predictions, _ = model(imgs)
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
        probe = "linear-probe" if args.blocks == "all" else "finetuned"
        track_loss(
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            lrates=lrates,
            save_dir=f"{model_path}/{probe}_training-curves.png"
        )
        # knn_sanity_check(model=model, loader=test_loader, device=device, savename=SAVENAME, epoch=epoch+1)
            
if __name__=="__main__":
    main()