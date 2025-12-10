import numpy as np 
import matplotlib.pyplot as plt 
import os 
import glob 
import tifffile 

def get_triplets(path: str):
    indices = [str(i) for i in range(26)]
    files = glob.glob(os.path.join(path, "*.tif"))
    triplets = [] 
    for index in indices:
        t = [
            os.path.join(path, f"sample_{index}_{modality}.tif") for modality in ["sted", "ddim", "draft"]
        ]
        triplets.append(t)
    return triplets

def save_triplets(triplets: list):
    for i, triplet in enumerate(triplets):
        print(f"[---] Triplet {i} of {len(triplets)} [---]")
        sted_path, ddim_path, draft_path = triplet 
        sted_img = tifffile.imread(sted_path)
        ddim_img = tifffile.imread(ddim_path)
        draft_img = tifffile.imread(draft_path)
        sted_img = np.clip(sted_img, 0, 1)
        ddim_img = np.clip(ddim_img, 0, 1)
        draft_img = np.clip(draft_img, 0, 1)
       
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(sted_img, cmap="hot")
        ax.axis("off")
        fig.savefig(f"./templates/sted_{i}.png", bbox_inches="tight", pad_inches=0, dpi=900)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(ddim_img, cmap="hot")
        ax.axis("off")
        fig.savefig(f"./candidates/ddim_sted_{i}.png", bbox_inches="tight", pad_inches=0, dpi=900)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(draft_img, cmap="hot")
        ax.axis("off")
        fig.savefig(f"./candidates/draft_sted_{i}.png", bbox_inches="tight", pad_inches=0, dpi=900)
        plt.close(fig)

if __name__=="__main__":
    triplets = get_triplets("./raw")
    save_triplets(triplets)
    
    
