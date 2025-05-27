import torch 
import random
import os
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import argparse 
import sys 

from stedfm.DEFAULTS import BASE_PATH
from stedfm.model_builder import get_pretrained_model_v2
from stedfm.loaders import get_dataset 
from stedfm.utils import compute_Nary_accuracy, update_cfg

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--global-pool", type=str, default='avg')
parser.add_argument("--blocks", type=str, default="all") # linear-probing by default
parser.add_argument("--pretraining", type=str, default="STED")
parser.add_argument("--probe", type=str, default="linear-probe", choices=["linear-probe", "finetuned"])
parser.add_argument("--from-scratch", action="store_true", default=False)
parser.add_argument("--dataset", type=str, default="synaptic-proteins")
parser.add_argument("--seeds", nargs="+", default=[])
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")    
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"
if len(args.seeds) == 1:
    args.seeds = args.seeds[0].split(" ")

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

def evaluate(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        probe : str
) -> float:
    model.eval()

    num_classes = loader.dataset.num_classes

    big_correct = np.array([0] * (num_classes+1))
    big_n = np.array([0] * (num_classes+1))
    with torch.no_grad():
        for imgs, data_dict in tqdm(loader, desc="Evaluation..."):
            labels = data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions, _ = model(imgs)

            correct, n = compute_Nary_accuracy(predictions, labels, N=num_classes)
            big_correct = big_correct + correct
            big_n = big_n + n
    accuracies = big_correct / big_n

    print(f"********* {args.model} - {args.pretraining} | {probe} | {args.dataset} **********")
    print("Overall accuracy = {:.3f}".format(accuracies[0]))
    for i in range(1, num_classes+1):
        acc = accuracies[i]
        print("Class {} accuracy = {:.3f}".format(
            i, acc))
    return accuracies[0]

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
    cfg.args = args
    update_cfg(cfg, args.opts)

    _, _, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        batch_size=cfg.batch_size,
        training=True,
        num_samples=None # not used when only loading test dataset
    )


    # Load state dict
    if len(args.seeds) > 0:
        accuracies = []
        for seed in args.seeds:
            modelname = args.model.replace("-lightning", "")
            model_path= os.path.join(BASE_PATH, "baselines", f"{modelname}_{SAVENAME}", args.dataset)
            model_name = f"{probe}-{seed}.pth"
            state_dict = torch.load(os.path.join(model_path, model_name), map_location="cpu")
            model.load_state_dict(state_dict["model_state_dict"])

            model = model.to(device)
            acc = evaluate(model=model, loader=test_loader, device=device, probe=probe)
            accuracies.append(acc)
        print("=====================================")
        print(f"Multiple repetitions with seeds: {args.seeds}")
        print(f"Model: {args.model}")
        print(f"Probe: {probe}")
        print(f"Dataset: {args.dataset}")
        print(f"Accuracies: {accuracies}")
        print(f"Mean accuracy: {np.mean(accuracies) * 100:0.2f}")
        print(f"Std accuracy: {np.std(accuracies) * 100:0.2f}")
        print("=====================================")
    else:        
        modelname = args.model.replace("-lightning", "")
        model_path= os.path.join(BASE_PATH, "baselines", f"{modelname}_{SAVENAME}", args.dataset)
        model_name = f"{probe}.pth"
        state_dict = torch.load(os.path.join(model_path, model_name), map_location="cpu")
        model.load_state_dict(state_dict["model_state_dict"])

        model = model.to(device)
        evaluate(model=model, loader=test_loader, device=device, probe=probe)
    

if __name__=="__main__":
    main()