import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import random 
import json 
from tqdm import tqdm, trange 
import argparse 
from loaders import get_dataset
import os
from model_builder import get_pretrained_model_v2
import tarfile
import io

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="peroxisome")
parser.add_argument("--num", type=int, default=9)
args = parser.parse_args()


def main():
    with tarfile.open("/home/frbea320/scratch/Datasets/JUMP_CP/jump.tar", "r") as tar:
        members = tar.getmembers()
        N = len(members)
        print(N)
        indices = random.sample(range(N), args.num)
        for i in indices:
            buffer = io.BytesIO()
            buffer.write(tar.extractfile(members[i]).read())
            buffer.seek(0)
            data = np.load(buffer, allow_pickle=True)
            img = data["image"]
            print(img.shape)
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis("off")
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            fig.savefig(f"jump_img_{i}.png", dpi=1200)
            plt.close(fig)


if __name__=="__main__":
    main()
