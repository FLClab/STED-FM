import torch 
import torchvision
from utils.data_utils import tar_dataloader 
import os
import numpy as np
from tqdm import tqdm 

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")
    means, vars = [], []
    dataloader = tar_dataloader(transform=None)
    for images in tqdm(dataloader, desc="Dataset stats..."):
        avg = torch.mean(images.squeeze(1), dim=(-2, -1))
        std = torch.std(images.squeeze(1), dim=(-2, -1))
        print(torch.mean(avg), torch.mean(std))
        means.append(torch.mean(avg).item())
        vars.append(torch.mean(std).item())

    sted_mean = np.mean(means)
    sted_var = np.mean(vars)

    print(f"Final mean: {sted_mean}")
    print(f"Final STD: {sted_var}")
    with open("./STED_dataset_stats.txt", "w") as handle:
        handle.writelines(['Mean', ' ', 'STD'])
        handle.writelines('\n')
        handle.writelines([str(sted_mean), ' ', str(sted_var)])

if __name__=="__main__":
    main()