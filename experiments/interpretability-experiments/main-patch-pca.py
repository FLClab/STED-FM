import numpy as np
import matplotlib.pyplot as plt
import torch 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import sys
import argparse
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from segmentation_datasets import get_dataset as get_actin_dataset
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH
from loaders import get_dataset 
from model_builder import get_pretrained_model_v2
sys.path.insert(0, "../segmentation-experiments")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
args = parser.parse_args()

def set_seeds():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

def compute_pca(data: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data)
    pca_per_image = np.split(pca_data, labels.shape[0])
    return pca_per_image, labels

def show_images(images: np.ndarray, pca_images: list, labels):
    for i, (og, img) in enumerate(zip(images, pca_images)):
        label = labels[i]
        img = MinMaxScaler().fit_transform(img).T
        img = torch.tensor(img, dtype=torch.float32)
        img = img.view(1, 3, 14, 14)
        img = F.interpolate(img, (224, 224), mode='bilinear').squeeze(0).cpu().data.numpy()

        if og.shape[0] == 3:
            og = og[0]
        # important to convert to numpy and then use np.transpose instead of staying as torch and using torch.view
        # funny business happening when using a combination of torch.view and F.interpolate --> does not give desired RGB image
        img = np.transpose(img, axes=(1, 2, 0)) 
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax.imshow(img)
        ax2.imshow(og, 'hot')
        ax.set_title(label)
        ax.axis("off")
        ax2.axis("off")
        fig.savefig(f"./patch-pca-examples/{args.dataset}-{args.weights}-pca-image-{i}.png", dpi=1200, bbox_inches="tight")
        plt.close(fig)

def main():
    SAVENAME = get_save_folder()
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if SAVENAME == "ImageNet" else 1

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if n_channels==3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="0",
        global_pool="patch", 
        num_classes=4
    )

    if args.dataset == "factin":
        _, _, test_dataset = get_actin_dataset(name=args.dataset, cfg=cfg)
        indices = np.arange(len(test_dataset))
        np.random.shuffle(indices)
        stop = 20
        counter = 0
        patch_embeddings = []
        og_images = []
        labels = [] # dummy
        with torch.no_grad():
            for i in indices:
                if counter >= stop:
                    break
                img, mask = test_dataset[i]
                if np.count_nonzero(mask[0]) and np.count_nonzero(mask[1]):
                    counter += 1
                    print(f"{counter}/{stop}")
                    img = img.unsqueeze(0)
                    embed = model.forward_features(img).squeeze(0).cpu().numpy()
                    patch_embeddings.append(embed)
                    labels.append(0)
                    if n_channels == 3:
                        og_images.append(img.cpu().numpy())
                    else:
                        og_images.append(img.squeeze(0).cpu().numpy())
    else:
        _, _, test_loader = get_dataset(name=args.dataset, training=True, n_channels=n_channels)
        test_dataset = test_loader.dataset 
        labels = test_dataset.labels 
        uniques = np.unique(labels)
        indices = []
        for u in uniques:
            cls_indices = np.where(labels == u)[0]
            cls_indices = np.random.choice(cls_indices, size=5, replace=False)
            indices.extend(cls_indices)

        patch_embeddings = []
        og_images = []
        labels = []
        with torch.no_grad():
            for i in indices:
                img, data_dict = test_dataset[i]
                label = data_dict["label"]
                img = img.unsqueeze(0) # B = 1
                embed = model.forward_features(img).squeeze(0).cpu().numpy()
                patch_embeddings.append(embed)
                labels.append(label)
                if n_channels == 3:
                    og_images.append(img.cpu().numpy())
                else:
                    og_images.append(img.squeeze(0).cpu().numpy())
            

    patch_embeddings = np.concatenate(patch_embeddings, axis=0)
    og_images = np.concatenate(og_images, axis=0)
    labels = np.array(labels)

    pca_per_image, labels = compute_pca(patch_embeddings, labels)
    show_images(og_images, pca_per_image, labels)

if __name__=="__main__":
    main()