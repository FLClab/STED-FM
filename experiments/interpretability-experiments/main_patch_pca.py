import numpy as np
import matplotlib.pyplot as plt
import torch 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import sys
import argparse
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from skimage.filters import threshold_otsu
from segmentation_datasets import get_dataset as get_actin_dataset
import os
from stedfm.DEFAULTS import BASE_PATH
from stedfm.loaders import get_dataset 
from stedfm.model_builder import get_pretrained_model_v2


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--threshold", type=str, default=None)
args = parser.parse_args()

FOREGROUND_THRESHOLD = 0.01

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
    elif "hpa" in args.weights.lower():
        return "HPA"
    elif "sim" in args.weights.lower():
        return "SIM"
    else:
        raise NotImplementedError("The requested weights do not exist.")

def compute_pca(data: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data)
    pca_per_image = np.split(pca_data, labels.shape[0])
    return pca_per_image, labels

def compute_foreground_pca(data: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> None:
    print(data.shape, masks.shape)
    pca_indices = np.where(masks == 1)[0]
    pca = PCA(n_components=3)
    new_data = np.zeros((data.shape[0], 3))
    pca_data = pca.fit_transform(data[pca_indices, :])
    new_data[pca_indices, :] = pca_data
    opposite_indices = np.setdiff1d(np.arange(data.shape[0]), pca_indices)
    new_data[opposite_indices, :] = (np.min(pca_data[:, 0]), np.min(pca_data[:, 1]), np.min(pca_data[:, 2]))
    pca_per_image = np.split(new_data, labels.shape[0])
    return pca_per_image, labels


def compute_pca_pca(data: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data)
    pca_mask = np.ones_like(pca_data)
    for i in range(pca_data.shape[0]):
        embed = pca_data[i, :]
        if embed[0] < 0.0:
            pca_data[i, :] = (np.nan, np.nan, np.nan)
    pca_per_image = np.split(pca_data, labels.shape[0])
    return pca_per_image, labels

def show_images(images: np.ndarray, pca_images: list, labels, save_folder: str):
    os.makedirs(f"./patch-pca-examples/{args.dataset}", exist_ok=True)

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
        fig.savefig(f"./patch-pca-examples/{args.dataset}/{save_folder}/{args.weights}-pca-image-{i}.pdf", dpi=1200, bbox_inches="tight")
        print(f"Saved image to ./patch-pca-examples/{args.dataset}/{save_folder}/{args.weights}-pca-image-{i}.pdf")
        plt.close(fig)

def show_foreground_images(images: np.ndarray, pca_images: list, masks: list, labels, save_folder: str):
    os.makedirs(f"./patch-pca-examples/{args.dataset}", exist_ok=True)
    for i, (og, img) in enumerate(zip(images, pca_images)):
        label = labels[i]
        img = MinMaxScaler().fit_transform(img).T
        img = torch.tensor(img, dtype=torch.float32)
        img = img.view(1, 3, 14, 14)
        img = F.interpolate(img, (224, 224), mode='bilinear').squeeze(0).cpu().data.numpy()

        mask = masks[i]
        print(mask.shape)
        mask = np.repeat(masks[i][:, :, np.newaxis], 3, axis=2)
        if og.shape[0] == 3:
            og = og[0]
        # important to convert to numpy and then use np.transpose instead of staying as torch and using torch.view
        # funny business happening when using a combination of torch.view and F.interpolate --> does not give desired RGB image
        img = np.transpose(img, axes=(1, 2, 0)) 
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax.imshow(img * mask)
        ax2.imshow(og, 'hot')
        ax.set_title(label)
        ax.axis("off")
        ax2.axis("off")
        fig.savefig(f"./patch-pca-examples/{args.dataset}/{save_folder}/{args.weights}-pca-image-{i}.pdf", dpi=1200, bbox_inches="tight")
        print(f"Saved image to ./patch-pca-examples/{args.dataset}/{save_folder}/{args.weights}-pca-image-{i}.pdf")
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
        pretrained=True if n_channels == 3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="all",
        global_pool="patch", 
        num_classes=4
    )
  
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
    patch_foreground_masks = []
    with torch.no_grad():
        for i in indices:
            img, data_dict = test_dataset[i]

            # For the foreground thresholding case
            img_numpy = img.squeeze(0).cpu().numpy()
            img_otsu = img_numpy > threshold_otsu(img_numpy)
            ys = np.arange(0, img_numpy.shape[0], 16)
            xs = np.arange(0, img_numpy.shape[1], 16)
            for y in ys:
                for x in xs:
                    patch = img_otsu[y:y+16, x:x+16]
                    foreground = np.count_nonzero(patch)
                    pixels = 16 * 16 
                    ratio = foreground / pixels
                    if ratio > FOREGROUND_THRESHOLD:
                        patch_foreground_masks.append(1)
                    else:
                        patch_foreground_masks.append(0)

            label = data_dict["label"]
            img = img.unsqueeze(0) # B = 1
            embed = model.forward_features(img).squeeze(0).cpu().numpy()
            patch_embeddings.append(embed)
            labels.append(label)
            if n_channels == 3:
                og_images.append(img.cpu().numpy())
            else:
                og_images.append(img.squeeze(0).cpu().numpy())
        
    if args.threshold is None:
        save_folder = "no-threshold"
        os.makedirs(f"./patch-pca-examples/{args.dataset}/{save_folder}", exist_ok=True)
        patch_embeddings = np.concatenate(patch_embeddings, axis=0)
        og_images = np.concatenate(og_images, axis=0)
        labels = np.array(labels)

        pca_per_image, labels = compute_pca(patch_embeddings, labels)
        show_images(og_images, pca_per_image, labels, save_folder=save_folder)

    elif args.threshold == "foreground":
        save_folder = "foreground-threshold"
        os.makedirs(f"./patch-pca-examples/{args.dataset}/{save_folder}", exist_ok=True) 
        patch_embeddings = np.concatenate(patch_embeddings, axis=0)
        patch_foreground_masks = np.array(patch_foreground_masks)
        og_images = np.concatenate(og_images, axis=0)
        labels = np.array(labels)

        pca_per_image, labels = compute_foreground_pca(patch_embeddings, labels, masks=patch_foreground_masks)
        show_images(og_images, pca_per_image, labels=labels, save_folder=save_folder)

    elif args.threshold == "pca":
        save_folder = "pca-threshold"
        os.makedirs(f"./patch-pca-examples/{args.dataset}/{save_folder}", exist_ok=True) 
        patch_embeddings = np.concatenate(patch_embeddings, axis=0)
        og_images = np.concatenate(og_images, axis=0)
        labels = np.array(labels)
        pca_per_image, labels = compute_pca_pca(patch_embeddings, labels)
        show_images(og_images, pca_per_image, labels, save_folder=save_folder)

    else:
        raise ValueError(f"Thresholding method '{args.threshold}' not recognized.")


if __name__=="__main__":
    main()