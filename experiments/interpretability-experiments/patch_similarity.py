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
import torch.nn.functional as F
from stedfm.DEFAULTS import BASE_PATH, COLORS
from stedfm.loaders import get_dataset 
from stedfm.model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--blocks", type=str, default="all")
parser.add_argument("--global-pool", type=str, default="avg")
parser.add_argument("--ckpt-path", type=str, default="")
parser.add_argument("--opts", nargs="+", default=[])
parser.add_argument("--figure", action="store_true")
parser.add_argument("--mode", type=str, default="dissimilar")
args = parser.parse_args()

def denormalize(img: torch.Tensor, mu: float = 0.0695771782959453, sigma: float = 0.12546228631005282) -> torch.Tensor:
    return img * sigma + mu
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

def display_examples(
        similarities: np.ndarray, 
        before_model: torch.nn.Module, 
        after_model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        n_samples: int = 10,
        mode: str = "dissimilar"
        ) -> None:
    os.makedirs(f"./patch-similarity/{mode}-examples", exist_ok=True)
    examples = np.argsort(similarities)
    examples = examples[:n_samples] if mode == "dissimilar" else examples[-n_samples:]
    for i in tqdm(examples, total=n_samples, desc=f"Displaying {mode} examples..."):
        img, label = dataset[i]
        img = img.unsqueeze(0).to(device)
        source_patches = before_model.backbone.forward_features(img)
        source_patches = source_patches[:, 1:, :]
        target_patches = after_model.backbone.forward_features(img)
        target_patches = target_patches[:, 1:, :]
        num_patches = target_patches.shape[1]
        heatmap = torch.zeros(num_patches)
        for j in range(num_patches):
            t_patch = target_patches[[0], j, :] 
            s_patch = source_patches[[0], j, :] 
            patch_similarity = torch.cosine_similarity(t_patch, s_patch, dim=1)
            patch_abnormality = 1 - patch_similarity 
            heatmap[j] = patch_abnormality
        final_heatmap = heatmap.view(14, 14).detach() 
        final_heatmap = F.interpolate(final_heatmap.view(1, 1, 14, 14), (224, 224), mode='bilinear').view(224, 224, 1)
        final_heatmap = final_heatmap.squeeze().detach().cpu().numpy()
        img = img.squeeze().detach().cpu().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        if "imagenet" in args.weights.lower():
            img = denormalize(img[0])
        axs[0].imshow(img, cmap='hot', vmin=0, vmax=1)
        axs[1].imshow(final_heatmap, cmap='viridis', vmin=0, vmax=1)
        axs[0].set_title(similarities[i])
        for ax in axs:
            ax.axis('off')
        plt.savefig(f"./patch-similarity/{mode}-examples/{args.weights}-example-{i}.png", dpi=1200, bbox_inches='tight')
        plt.close(fig)

    
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
            image_similarities = []
            all_similarities = []
            for i in trange(N):
                img, label = test_dataset[i]
                img = img.unsqueeze(0).to(DEVICE)
                source_patches = before_model.backbone.forward_features(img)
                target_patches = after_model.backbone.forward_features(img)
                sim = compute_patch_similarity(source_patches, target_patches)
                image_sim = np.min(sim) if args.mode == "dissimilar" else np.max(sim)
                image_similarities.append(image_sim)
                all_similarities.extend(sim)


        all_similarities = np.array(all_similarities)
        image_similarities = np.array(image_similarities)
        display_examples(image_similarities, before_model, after_model, test_dataset, DEVICE, mode=args.mode)
        exit()
        np.savez(f"./patch-similarity/similarity-scores/{args.weights}-similarity-hist.npz", similarities=all_similarities)
        plt.hist(all_similarities, bins=100, color=COLORS[get_save_folder()])
        plt.xlabel("Cosine similarity")
        plt.ylabel("Frequency")
        plt.savefig(f"./patch-similarity/similarity-scores/{args.weights}-similarity-hist.png")

            

if __name__=="__main__":
    main()



