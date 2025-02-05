import numpy as np 
import tarfile 
import io
from tqdm import tqdm
import argparse 
import h5py 
from model_builder import get_base_model
import sys 
sys.path.insert(0, "./segmentation-experiments")
from datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="250k")
args = parser.parse_args()

def main_footprocess():
    model, cfg = get_base_model("mae-lightning-small")
    dataset, _, _ = get_dataset(args.dataset, cfg=cfg)
    data_means, data_stds = [], []
    for img, _ in dataset:
        img = img.numpy()
        data_means.append(np.mean(img))
        data_stds.append(np.std(img))

    print("=== Dataset footprocess stats ===\n")
    print(f"\tMean: {np.mean(data_means)}")
    print(f"\tStd: {np.mean(data_stds)}")

def main_lioness():
    model, cfg = get_base_model("mae-lightning-small")
    dataset, _, _ = get_dataset(args.dataset, cfg=cfg)
    data_means, data_stds = [], []
    for img, _ in dataset:
        img = img.numpy()
        data_means.append(np.mean(img))
        data_stds.append(np.std(img))

    print("=== Dataset lioness stats ===\n")
    print(f"\tMean: {np.mean(data_means)}")
    print(f"\tStd: {np.mean(data_stds)}")

def main_250k():
    data_means, data_stds = [], []
    with tarfile.open("/home-local/Frederic/Datasets/FLCDataset/dataset-250k.tar", "r") as handle:
        members = handle.getmembers()
        N = len(members)
        for member in tqdm(members, total=N):
            buffer = io.BytesIO()
            buffer.write(handle.extractfile(member).read())
            buffer.seek(0)
            data = np.load(buffer, allow_pickle=True)
            img = data["image"] / 255.
            data_means.append(np.mean(img))
            data_stds.append(np.std(img))

    print("=== Dataset 250k stats ===\n")
    print(f"\tMean: {np.mean(data_means)}")
    print(f"\tStd: {np.mean(data_stds)}")

def main_resolution():
    data_means, data_stds = [], []
    with h5py.File("/home-local/Frederic/evaluation-data/low-high-quality/training.hdf5", "r") as handle:
        high_images = handle["high/rabBassoon STAR635P"][()]
        low_images = handle["low/rabBassoon STAR635P"][()]
        images = np.concatenate([high_images, low_images], axis=0)
        for img in images:
            data_means.append(np.mean(img))
            data_stds.append(np.std(img))

    print("=== Dataset resolution stats ===\n")
    print(f"\tMean: {np.mean(data_means)}")
    print(f"\tStd: {np.mean(data_stds)}")

if __name__=="__main__":
    if args.dataset == "250k":
        main_250k()
    elif args.dataset == "resolution":
        main_resolution()
    elif args.dataset == "lioness":
        main_lioness()
    elif args.dataset == "footprocess":
        main_footprocess() 
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
