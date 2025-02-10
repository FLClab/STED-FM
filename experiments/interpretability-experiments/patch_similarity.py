import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import argparse 
from tqdm import tqdm, trange 
import os 
import glob
import sys 
from scipy.stats import gaussian_kde
import random
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH, COLORS
from loaders import get_dataset 
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--blocks", type=str, default="all")
parser.add_argument("--global-pool", type=str, default="avg")
parser.add_argument("--ckpt-path", type=str, default="")
parser.add_argument("--opts", nargs="+", default=[])
parser.add_argument("--samples-per-class", type=int, default=20)
parser.add_argument("--figure", action="store_true")
args = parser.parse_args()

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
    elif "sim" in args.weights.lower():
        return "SIM"        
    elif "jump" in args.weights.lower():
        return "JUMP"
    elif "ctc" in args.weights.lower():
        return "CTC"
    elif "hpa" in args.weights.lower():
        return "HPA"
    elif "sim" in args.weights.lower():
        return "SIM"
    elif "hybrid" in args.weights.lower():
        return "Hybrid"
    else:
        raise NotImplementedError("The requested weights do not exist.")
    
def compute_patch_similarity(source_patches, target_patches):
    feature_sims = []
    num_patches = source_patches.shape[1]
    for i in range(num_patches):
        sp = source_patches[[0], i, :]
        tp = target_patches[[0], i, :]
        sim = torch.cosine_similarity(sp, tp, dim=1).item()
        feature_sims.append(sim)
    return feature_sims

def load_data(path: str):
    results = {}
    files = glob.glob(os.path.join(path, "*.npz"))
    for file in files:
        pretraining = file.split("/")[-1].split("-")[0].split("_")
        pretraining = "_".join(pretraining[2:])
        data = np.load(file)["similarities"]
        results[pretraining] = data
    return results


def ridgeline(data, overlap=0, fill=True, labels=None, n_points=500):
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(0, 1, n_points)
    curves = []
    ys = []
    max_density = []
    for i, (key, value) in enumerate(data.items()):
        pdf = gaussian_kde(value)
        y = i*(1.0-overlap)
        ys.append(y)
        curve = pdf(xx)
        curve = (curve - curve.min()) / (curve.max() - curve.min())
        idx = np.argmax(curve)
        max_density.append(xx[idx])
        curves.append(y + curve[idx])
       
        plt.fill_between(xx, np.ones(n_points)*y, curve+y, zorder=len(data)-i+1, color=COLORS[key])
        plt.plot(xx, curve+y, c='white', zorder=len(data)-i+1)
    if labels:
        plt.yticks(ys, labels)
    return max_density

    
def plot_ridgelines(data, labels: list) -> None:
    fig = plt.figure(figsize=(8, 10))

    max_density = ridgeline(data, labels=labels, overlap=.40, fill='tomato')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel('Cosine similarity')
    plt.xlim([0.0, 1.0])
    plt.grid(zorder=0)
    # fig.savefig(f"./ridgeline_plots/theresa_1/{args.model}_{key}_ridgeline_{num}events.png")
    fig.savefig(f"./patch-similarity/similarity-scores/similarity-ridgeline.png", bbox_inches='tight')
    return max_density


    
def main():
    if args.figure:
        results = load_data("./patch-similarity/similarity-scores")
        results = {k: results[k] for k in ["IMAGENET1K_V1", "JUMP", "HPA", "SIM", "STED"]}
        plot_ridgelines(results, labels=list(results.keys()))
    else:
        os.makedirs(f"./patch-similarity/similarity-scores", exist_ok=True)
        set_seeds()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_channels = 3 if "imagenet" in args.weights.lower() else 1
        before_model, cfg = get_pretrained_model_v2(
            name=args.model,
            weights=args.weights,
            path=None,
            mask_ratio=0.0,
            pretrained=True if n_channels==3 else False,
            in_channels=n_channels,
            as_classifier=True,
            blocks=args.blocks,
            num_classes=4,
            from_scratch=False
        )
        after_model, _ = get_pretrained_model_v2(
            name=args.model,
            weights=args.weights,
            path=None,
            mask_ratio=0.0,
            pretrained=True if n_channels==3 else False,
            in_channels=n_channels,
            as_classifier=True,
            blocks=args.blocks,
            num_classes=4,
            from_scratch=False
        )
        ckpt = torch.load(args.ckpt_path)
        after_model.load_state_dict(ckpt["model_state_dict"])
        before_model.eval()
        after_model.eval()
        before_model = before_model.to(DEVICE)
        after_model = after_model.to(DEVICE)

        _, _, test_loader = get_dataset(name=args.dataset, training=True, n_channels=n_channels)
        test_dataset = test_loader.dataset 
        labels = test_dataset.labels 
        
        N = len(test_dataset)

        with torch.no_grad():
            all_similarities = []
            for i in trange(N):
                img, label = test_dataset[i]
                img = img.unsqueeze(0).to(DEVICE)
                source_patches = before_model.backbone.forward_features(img)
                target_patches = after_model.backbone.forward_features(img)
                sim = compute_patch_similarity(source_patches, target_patches)
                all_similarities.extend(sim)


        all_similarities = np.array(all_similarities)
        np.savez(f"./patch-similarity/similarity-scores/{args.weights}-similarity-hist.npz", similarities=all_similarities)
        plt.hist(all_similarities, bins=100, color=COLORS[get_save_folder()])
        plt.xlabel("Cosine similarity")
        plt.ylabel("Frequency")
        plt.savefig(f"./patch-similarity/similarity-scores/{args.weights}-similarity-hist.png")

            

if __name__=="__main__":
    main()



