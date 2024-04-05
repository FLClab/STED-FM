import h5py
import numpy as np
from typing import List
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib import colors


PATH = "/home/frederic/Datasets/FLCDataset/theresa_proteins.hdf5"

gray_cm = plt.get_cmap('gray', 256)
red_colors = gray_cm(np.linspace(0, 1, 256))
red_colors[:, [1,2]] = 0.0
red_cm = colors.ListedColormap(red_colors)

green_colors = gray_cm(np.linspace(0, 1, 256))
green_colors[:, [0,2]] = 0.0
green_cm = colors.ListedColormap(green_colors)

magenta_colors = gray_cm(np.linspace(0, 1, 256))
magenta_colors[:, 1] = 0.0
magenta_cm = colors.ListedColormap(magenta_colors)

cyan_colors = gray_cm(np.linspace(0, 1, 256))
cyan_colors[:, 0] = 0.0
cyan_cm = colors.ListedColormap(cyan_colors)

def load_images(path: str = PATH):
    found = [0 ,0, 0, 0, 0]
    images = []
    with h5py.File(PATH, "a") as hf:
        N = int(hf["protein"].size)
        labels = hf["protein"][()]
        print(np.unique(labels, return_counts=True))
        indices = np.arange(0, N, 1)
        np.random.shuffle(indices)
        for i in indices:
            if sum([item == 1 for item in found]) == 4:
                break
            img = hf["images"][i]
            protein = int(hf["protein"][i])
            if protein > 1:
                protein -= 1 # Because we removed NKCC (label = 2) from the dataset
            if found[protein] != 0:
                continue
            else:
                images.append(img)
                found[protein] += 1
    return images


def save_examples(images: List[np.ndarray]):
    colormaps = [red_cm, magenta_cm, cyan_cm, green_cm]
    proteins = ["Bassoon", 'Homer', 'Rim', "PSD95"]
    for img, protein, cmap in zip(images, proteins, colormaps):
        fig = plt.figure()
        plt.imshow(img, cmap=cmap, vmax=img.max() * 0.6)
        plt.xticks([])
        plt.yticks([])
        plt.title(protein)
        fig.savefig(f"./{protein}_5.png", dpi=1200, bbox_inches='tight')
        plt.close(fig)

def main():
    images = load_images()
    np.random.shuffle(images)
    save_examples(images)

if __name__=="__main__":
    main()