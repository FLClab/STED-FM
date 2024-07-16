import numpy as np
import h5py
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

OUTPATH = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/zooniverse/"

def get_files():
    path = f"{OUTPATH}/raw" 
    files = glob.glob(f"{path}/*.npz")
    files = list(set(files))
    return files

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def main():
    files = get_files()
    crop_size = 224

    with h5py.File(f"{OUTPATH}/train.hdf5", "a") as train_handle:
        with h5py.File(f"{OUTPATH}/valid.hdf5", "a") as valid_handle:
            with h5py.File(f"{OUTPATH}/test.hdf5", "a") as test_handle:
                train_imgs = train_handle.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
                valid_imgs = valid_handle.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
                test_imgs = test_handle.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))

                train_masks = train_handle.create_dataset(name="masks", shape=(0, 6, 224, 224), maxshape=(None, 6, 224, 224))
                valid_masks = valid_handle.create_dataset(name="masks", shape=(0, 6, 224, 224), maxshape=(None, 6, 224, 224))
                test_masks = test_handle.create_dataset(name="masks", shape=(0, 6, 224, 224), maxshape=(None, 6, 224, 224))

              

                for f in tqdm(files):
                    data = np.load(f)
                    img, mask = data['img'], data["semantic_mask"]
                    img = normalize(img)
                    fig, axs = plt.subplots(1, 7, figsize=(20, 5))
                    axs[0].imshow(img, cmap='hot', vmax=1.0)
                    axs[1].imshow(mask[0], cmap='gray')
                    axs[2].imshow(mask[1], cmap='gray')
                    axs[3].imshow(mask[2], cmap='gray')   
                    axs[4].imshow(mask[3], cmap='gray')
                    axs[5].imshow(mask[4], cmap='gray')
                    axs[6].imshow(mask[5], cmap='gray')
                    for t, ax in zip(["img", "round", "elongated", "multidomain", "irregular", "perforated", "noise"], axs):
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(t)
                    plt.tight_layout()
                    fig.savefig(f"{OUTPATH}/temp.pdf", bbox_inches='tight', dpi=1200)

                    
                    Y, X = img.shape
                    ys = np.arange(0, Y - crop_size, crop_size)
                    xs = np.arange(0, X - crop_size, crop_size)
                    for y in ys:
                        for x in xs:
                            crop = img[y:y+crop_size, x:x+crop_size]
                            mask_crop = mask[:, y:y+crop_size, x:x+crop_size]

                            set_prob = random.random()
                            if set_prob <= 0.80:
                                train_imgs.resize(train_imgs.shape[0] + 1, axis=0)
                                train_masks.resize(train_masks.shape[0] + 1, axis=0)
                                train_imgs[-1:] = crop
                                train_masks[-1:] = mask_crop

                            elif 0.80 < set_prob <= 0.90:
                                valid_imgs.resize(valid_imgs.shape[0] + 1, axis=0)
                                valid_masks.resize(valid_masks.shape[0] + 1, axis=0)
                                valid_imgs[-1:] = crop
                                valid_masks[-1:] = mask_crop

                            else:
                                test_imgs.resize(test_imgs.shape[0] + 1, axis=0)
                                test_masks.resize(test_masks.shape[0] + 1, axis=0)
                                test_imgs[-1:] = crop
                                test_masks[-1:] = mask_crop



if __name__=="__main__":
    main()