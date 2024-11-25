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
from skimage.filters import threshold_otsu
import copy
import os
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
parser.add_argument("--threshold", type=str, default=None)
parser.add_argument("--patch-size", type=int, default=16)
parser.add_argument("--image-size", type=int, default=224)
args = parser.parse_args()

def set_seeds():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def threshold_folder():
    if args.threshold is None:
        return "all-patches"
    elif args.threshold == "foreground":
        return "foreground-patches"
    elif args.threshold == "pca1":
        return "pca1-patches"
    else:
        raise NotImplementedError

THRESHOLD_FOLDER = threshold_folder()

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
    print(pca_data.shape)
    if args.threshold is None:
        for ch in range(pca_data.shape[1]):
            m, M = pca_data[:, ch].min(), pca_data[:, ch].max()
            pca_data[:, ch] = (pca_data[:, ch] - m) / (M - m)

    elif args.threshold == "pca1":
        indices = np.where(pca_data[:, 0] >= 0)[0]
        mask = pca_data[:, 0] >= 0
        assert np.all(pca_data[indices, 0] >=0)
        for ch in range(pca_data.shape[1]):
            m, M = pca_data[indices, ch].min(), pca_data[indices, ch].max()
            pca_data[indices, ch] = (pca_data[indices, ch] - m) / (M - m) 

    else:
        pass

    pca_per_image = np.reshape(pca_data, (20, 196, 3))
    pca_mask = np.split(mask, labels.shape[0])

    return pca_per_image, pca_mask, labels

def get_foreground_mask(img: np.ndarray, threshold: float = 0.25):
    pixel_count = args.patch_size ** 2
    mask = np.zeros((args.image_size, args.image_size, 3))
    ys = np.arange(0, args.image_size, args.patch_size)
    xs = np.arange(0, args.image_size, args.patch_size)
    for y in ys:
        for x in xs:
            patch = img[y:y+args.patch_size, x:x+args.patch_size]
            patch_tau = patch >= threshold_otsu(patch)
            foreground = np.count_nonzero(patch_tau)
            ratio = foreground / pixel_count
            if ratio >= threshold:
                mask[y:y+args.patch_size, x:x+args.patch_size] = np.ones((args.patch_size, args.patch_size, 3))
    return mask

def get_pca_mask(img):
    img = img.T
    img = torch.tensor(img, dtype=torch.float32)
    img = img.view(1, 3, 14, 14)
    img = img.squeeze(0).cpu().data.numpy()
    img = np.transpose(img, axes=(1,2,0))
   
    assert img.shape == (14, 14, 3)
    ys = np.arange(0, 224, 16)
    xs = np.arange(0, 224, 16)
    mask = np.zeros((args.image_size, args.image_size, 3))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j][0] >= 0:
                y = int(ys[i])
                x = int(xs[j])
                mask[y:y+args.patch_size, x:x+args.patch_size] = np.ones((args.patch_size, args.patch_size, 3))
    return mask

def interpolate_mask(mask):
    ys = np.arange(0, 224, 16)
    xs = np.arange(0, 224, 16)
    intrp_mask = np.zeros((args.image_size, args.image_size, 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                y = int(ys[i])
                x = int(xs[j])
                intrp_mask[y:y+args.patch_size, x:x+args.patch_size] = np.ones((args.patch_size, args.patch_size, 3))
    return intrp_mask


def show_images(images: np.ndarray, pca_images: list, pca_mask: list, labels):
    if not os.path.exists(f"./patch-pca-examples/{THRESHOLD_FOLDER}/{args.dataset}"):
        print(f"--- Creating directory for {args.dataset} results ---")
        os.mkdir(f"./patch-pca-examples/{THRESHOLD_FOLDER}/{args.dataset}")

    for i, (og, img, mask) in enumerate(zip(images, pca_images, pca_mask)):
        label = labels[i]
        img_temp = copy.deepcopy(img)
        # img = MinMaxScaler().fit_transform(img).T
        img = img.T
        mask = mask.T
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        mask = mask.view(14, 14).detach().cpu().numpy() * 1.0
        print(mask)

        img = img.view(1, 3, 14, 14)
        img = F.interpolate(img, (224, 224), mode='bilinear').squeeze(0).cpu().data.numpy()

        if args.threshold is None:
            m = np.ones((224, 224, 3))
        elif args.threshold == "foreground":
            m = get_foreground_mask(img=og)
        elif args.threshold == "pca1":
            print("--- Interpolating patch mask ---")
            m = interpolate_mask(mask)
        else: 
            raise NotImplementedError


        if og.shape[0] == 3:
            og = og[0]
        # important to convert to numpy and then use np.transpose instead of staying as torch and using torch.view
        # funny business happening when using a combination of torch.view and F.interpolate --> does not give desired RGB image
        img = np.transpose(img, axes=(1, 2, 0)) 
        img = img * m
      
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax.imshow(img)
        ax2.imshow(og, 'hot')
        ax.set_title(label)
        ax.axis("off")
        ax2.axis("off")
        fig.savefig(f"./patch-pca-examples/{THRESHOLD_FOLDER}/{args.dataset}/{args.weights}-pca-image-{i}.png", dpi=1200, bbox_inches="tight")
        plt.close(fig)
        exit()

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

    pca_per_image, pca_mask, labels = compute_pca(patch_embeddings, labels)
    show_images(og_images, pca_per_image, pca_mask, labels)

if __name__=="__main__":
    main()