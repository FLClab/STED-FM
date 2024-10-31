import numpy as np
import matplotlib.pyplot as plt
import torch 
import random
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse 
import sys 
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from lightly.utils.scheduler import CosineWarmupScheduler

try:
    from torchinfo import summary
except ModuleNotFoundError:
    from torchsummary import summary
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH
from loaders import get_dataset
from model_builder import get_pretrained_model_v2 
from utils import SaveBestModel, AverageMeter, ScoreTracker, compute_Nary_accuracy, track_loss_steps, update_cfg, get_number_of_classes

plt.style.use("dark_background")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default='synaptic-proteins')
parser.add_argument("--model", type=str, default='mae-lightning-small')
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--global-pool", type=str, default='avg')
parser.add_argument("--blocks", type=str, default="all") # linear-probing by default
parser.add_argument("--from-scratch", action="store_true", 
                    help="Activates the `from-scratch` training mode")
parser.add_argument("--track-epochs", action="store_true")
parser.add_argument("--num-per-class", type=int, default=None)
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")    
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

def set_seeds():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_save_folder() -> str: 
    if args.weights is None:
        return "from-scratch"
    elif "imagenet" in args.weights.lower():
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
            print(votes)
            confusion_matrix[unique, predicted_unique] += np.sum(votes >= 3)
    accuracy = np.diag(confusion_matrix).sum() / np.sum(confusion_matrix)
    print(f"--- Epoch {epoch} --> {args.dataset} ; {args.model} ; {savename} ---\n\tAccuracy: {accuracy * 100:0.2f}\n")

def plot_features(features, labels, savename, classes=None, **kwargs):
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(4, 4))
    if classes is not None:
        uniques = np.unique(labels)
        cmap = plt.get_cmap("rainbow", len(classes))
        for i, unique in enumerate(uniques):
            mask = labels == unique
            ax.scatter(pca_features[mask, 0], pca_features[mask, 1], color=cmap(i), label=classes[i])
        ax.legend()
    else:
        ax.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap="rainbow")
    ax.set(
        ylabel="PCA-2", xlabel="PCA-1"
    )
    if isinstance(savename, str):
        fig.savefig(savename, bbox_inches="tight")
    plt.close()
    
def validation_step(model, valid_loader, criterion, epoch, device, save_dir=None, **kwargs):
    model.eval()
    loss_meter = AverageMeter()

    num_classes = valid_loader.dataset.num_classes

    correct, N = np.array([0] * (num_classes+1)), np.array([0] * (num_classes+1))
    all_features, all_labels = [], []
    with torch.no_grad():
        for imgs, data_dict in tqdm(valid_loader, desc="Validation...", leave=False):
            labels =  data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions, features = model(imgs)

            loss = criterion(predictions, labels)
            loss_meter.update(loss.item())
            c, n = compute_Nary_accuracy(predictions, labels, N=num_classes)
            correct = correct + c 
            N = n + N
            all_features.extend(features.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

    accuracies = correct / N 
    print("********* Validation metrics **********")
    print("Epoch {} validation loss = {:.3f} ({:.3f})".format(
        epoch + 1, loss_meter.val, loss_meter.avg))
    print("Overall accuracy = {:.3f}".format(accuracies[0]))
    for i in range(1, num_classes+1):
        acc = accuracies[i]
        print("Class {} accuracy = {:.3f}".format(
            i, acc))  
    # plot_features(all_features, all_labels, savename=save_dir, **kwargs)

    return loss_meter.avg, accuracies[0]


def main():
    # set_seeds()
    # num_classes = get_number_of_classes(dataset=args.dataset)
    train_loader, _, _ = get_dataset(
        name=args.dataset, training=True
    )
    num_classes = train_loader.dataset.num_classes
    print("=====================================")
    print(f"Dataset: {args.dataset}")
    print(f"Num. Classes: {num_classes}")
    print(f"Classes: {train_loader.dataset.classes}")
    print("=====================================")

    set_seeds()

    SAVENAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if SAVENAME == "ImageNet" else 1

    probe = "linear-probe" if args.blocks == "all" else "finetuned"
    if args.from_scratch:
        probe = "from-scratch"
        args.weights = None
        args.blocks = "0"

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if n_channels==3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks=args.blocks,
        num_classes=num_classes
    )
    model = model.to(device)

    # Update configuration
    cfg.args = args
    update_cfg(cfg, args.opts)

    # summary(model, input_size=(1, 224, 224), device=device.type)

    train_loader, valid_loader, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        batch_size=cfg.batch_size,
        training=True,
        num_samples=args.num_per_class,
    )
    
    #num_epochs = 300
    budget = train_loader.dataset.original_size * 300
    num_epochs = int(budget / (args.num_per_class * train_loader.dataset.num_classes))
    warmup_epochs = 0.1 * num_epochs
    print(f"--- Training with {args.num_per_class} samples per class for {num_epochs} epochs ---")

    if probe == "from-scratch":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup_epochs=warmup_epochs, max_epochs=num_epochs,
            start_value=1.0, end_value=0.01
        )
    elif probe == "linear-probe":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup_epochs=warmup_epochs, max_epochs=num_epochs,
            start_value=1.0, end_value=0.01,
            period=num_epochs//10
        )        
        # scheduler = CosineAnnealingLR(
        #     optimizer=optimizer, T_max=num_epochs/10, eta_min=1e-5
        # )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) 
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup_epochs=warmup_epochs, max_epochs=num_epochs,
            start_value=1.0, end_value=0.01
        )

    criterion = torch.nn.CrossEntropyLoss()
    modelname = args.model.replace("-lightning", "")
    model_path= os.path.join(BASE_PATH, "baselines", f"{modelname}_{SAVENAME}", args.dataset)
    #model_path = f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{modelname}_{SAVENAME}/{args.dataset}"
    os.makedirs(model_path, exist_ok=True)

    # Training loop
    train_loss, val_loss, val_acc, lrates = ScoreTracker(), ScoreTracker(), ScoreTracker(), ScoreTracker()
    save_best_model = SaveBestModel(
        save_dir=f"{model_path}",
        model_name=f"{probe}_{args.num_per_class}_{args.seed}"
    )

    # knn_sanity_check(model=model, loader=test_loader, device=device, savename=SAVENAME, epoch=0)

    step = 0
    for epoch in trange(num_epochs, desc="Epochs..."):
        model.train()
        loss_meter = AverageMeter()
        for imgs, data_dict in tqdm(train_loader, desc="Training...", leave=False):
            labels = data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            predictions, _ = model(imgs)
            _, preds = torch.max(predictions, 1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())


            if step % 100 == 0:
                v_loss, v_acc = validation_step(
                    model=model,
                    valid_loader=valid_loader,
                    criterion=criterion,
                    epoch=epoch,
                    device=device,
                    save_dir = f"{save_best_model.save_dir}/{save_best_model.model_name}_pca.png",
                    classes = train_loader.dataset.classes
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

                val_loss.update(step, v_loss)
                val_acc.update(step, v_acc)
                
            step += 1

        temp_lr = optimizer.param_groups[0]['lr']
        lrates.update(step, temp_lr)
        train_loss.update(step, loss_meter.avg)

        scheduler.step()        
        track_loss_steps(
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            lrates=lrates,
            save_dir=f"{save_best_model.save_dir}/{save_best_model.model_name}-{args.num_per_class}_training-curves.png"
        )
        # knn_sanity_check(model=model, loader=test_loader, device=device, savename=SAVENAME, epoch=epoch+1)

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if n_channels==3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks=args.blocks,
        num_classes=num_classes
    )
    state_dict = torch.load(f"{save_best_model.save_dir}/{save_best_model.model_name}.pth", map_location="cpu")
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(device)

    loss, acc = validation_step(
        model=model,
        valid_loader=test_loader,
        criterion=criterion,
        epoch=num_epochs,
        device=device
    )
    print("=====================================")
    print(f"Dataset: {args.dataset}")
    print(f"Testing loss: {loss}")
    print(f"Testing accuracy: {acc * 100:0.2f}")
    print("=====================================")

if __name__=="__main__":
    main()