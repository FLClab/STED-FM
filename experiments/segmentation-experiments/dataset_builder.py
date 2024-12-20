import numpy as np
import h5py
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse 
import tarfile 
from typing import List
import io
import tifffile
from wavelet import detect_spots

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/home-local/Frederic/Datasets/Neural-Activity-States/PSD95-Basson")
parser.add_argument("--crop-size", type=int, default=224)
parser.add_argument("--foreground-threshold", type=int, default=0.001)
parser.add_argument("--num-files", type=int, default=500)
args = parser.parse_args()

def load_image_files(dataset_path: str) -> List[str]:
    image_files = glob.glob(f"{dataset_path}/**/*.tif", recursive=True)
    image_files = list(set(image_files))
    return image_files

def save_image(img: np.ndarray, mask: np.ndarray):
    fig, axs = plt.subplots(1, 2, figsize=(7,5))
    axs[0].imshow(img, cmap="hot", vmin=0.0, vmax=1.0)
    axs[1].imshow(mask, cmap="gray")
    for ax in axs:
        ax.axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    plt.savefig("temp.png", dpi=1200)
    plt.close(fig)


def main():
    image_files = load_image_files(args.dataset_path)
    counter = 0
    with tarfile.open(f"{args.dataset_path}/synaptic-protein-segmentation.tar", "a") as handle:
        pbar = tqdm(image_files)
        for f in pbar:
            ch = np.random.randint(2)
            protein = ["PSD95", "Bassoon"][ch]
            img = tifffile.imread(f)[ch]
            m, M = np.quantile(img, 0.001), np.quantile(img, 0.999)
            img = (img - m) / (M - m)
            img = np.clip(img, 0.0, 1.0)
            mask = detect_spots(img)

            num_y = np.floor(img.shape[0] / args.crop_size)
            num_x = np.floor(img.shape[1] / args.crop_size)
            ys = np.arange(0, num_y * args.crop_size, args.crop_size).astype("int")
            xs = np.arange(0, num_x * args.crop_size, args.crop_size).astype("int")
            for y in ys:
                for x in xs:
                    if counter >= args.num_files:
                        print("=== Done, saved 200 files to tarball ===")
                        exit()
                    crop = img[y:y+args.crop_size, x:x+args.crop_size]
                    mask_crop = mask[y:y+args.crop_size, x:x+args.crop_size]

                    assert crop.shape == mask_crop.shape
                    assert crop.shape == (args.crop_size, args.crop_size)
                    # save_image(crop, mask_crop)
                    foreground = np.count_nonzero(mask_crop) 
                    ratio = foreground / (args.crop_size**2) 
                    if ratio <= args.foreground_threshold:
                        continue
                    else:
                        counter += 1
                        pbar.set_description(f"Saved {counter} files to tarball")
                        buffer = io.BytesIO()
                        np.savez(
                            file=buffer,
                            img=crop,
                            segmentation=mask_crop,
                        )
                        buffer.seek(0)
                        tarinfo = tarfile.TarInfo(f"{counter}-{protein}")
                        tarinfo.size = len(buffer.getbuffer())
                        handle.addfile(tarinfo=tarinfo, fileobj=buffer)

    print(f"=== Done, saved {counter} files to tarball ===")
if __name__=="__main__":
    main()