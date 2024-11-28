import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import random 
import json 
from tqdm import tqdm, trange 
import argparse 
from loaders import get_dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="peroxisome")
parser.add_argument("--num", type=int, default=20)
args = parser.parse_args()


def main():
    if not os.path.exists(f"./dummy/{args.dataset}"):
        os.mkdir(f"./dummy/{args.dataset}")

    dataloader, _, _ = get_dataset(args.dataset)
    dataset = dataloader.dataset 
    print(dataset.classes)
    N = len(dataset)
    indices = np.random.choice(np.arange(N), size=args.num)

    for counter, idx in enumerate(indices):
        img, data_dict = dataset[idx]
        img = img.squeeze(0).detach().cpu().numpy()
        label = data_dict["label"]
        fig = plt.figure()
        plt.imshow(img, cmap='hot')
        plt.axis("off")
        plt.title(label)
        fig.savefig(f"./dummy/{args.dataset}/sample_{counter}.png", bbox_inches='tight', dpi=1200)
        plt.close(fig)
        if counter >= 20:
            exit()

        
        


if __name__=="__main__":
    main()
