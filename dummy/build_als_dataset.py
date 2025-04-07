import numpy as np 
import matplotlib.pyplot as plt 
import tifffile
import argparse 
import os 
import glob
import tarfile 
from tqdm import tqdm
from typing import List
from skimage import measure
import io
import sys 
sys.path.insert(0, "../../Neurodegeneration/")
from wavelet import DetectionWavelets

THRESHOLD = 0.03

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default="/home-local/Frederic/Datasets/ALS/ALS_JM_Fred_unmixed")
parser.add_argument("--batch-id", type=str, default="2023-03-01 (26.2) (dendrites 4c Batch3)")
parser.add_argument("--condition", type=str, default="PLKO")
parser.add_argument("--channel", type=str, default="PSD95")
args = parser.parse_args()

def get_metadata(filename: str):
    folder = filename.split("/")[-3]
    div, condition, dpi = folder.split(" ")
    div = div.replace("INF", "")
    metadata = (condition, div, dpi)
    return metadata

def load_files(path: str, batch_id: str, condition: str):
    files = glob.glob(f"{path}/{batch_id}/**/**/*tif", recursive=True)
    files = list(set(files))
    files = [f for f in files if condition in f]
    N = len(files)
    train_files = np.random.choice(files, size=int(0.6*N), replace=False)
    valid_files = [f for f in files if f not in train_files]
    test_files = np.random.choice(valid_files, size=int(0.5*len(valid_files)), replace=False)
    train_files = [f for f in train_files if f not in test_files]
    valid_files = [f for f in valid_files if f not in test_files]
    assert not np.any([f in test_files for f in train_files])
    assert not np.any([f in train_files for f in valid_files])
    assert not np.any([f in valid_files for f in test_files])
    
    # train_metadata = [get_metadata(f) for f in train_files] 
    # train_divs = [m[1] for m in train_metadata]
    # valid_metadata = [get_metadata(f) for f in valid_files]
    # valid_divs = [m[1] for m in valid_metadata]
    # test_metadata = [get_metadata(f) for f in test_files]
    # test_divs = [m[1] for m in test_metadata]
    return train_files, valid_files, test_files

def process_files(files: List[str], output_path: str):
    added_files = []
    num_crops = 0
    channel = 0 if args.channel == "FUS" else 2
    with tarfile.open(f"{output_path}.tar", "a") as handle:
        for f in tqdm(files, desc="Processing files..."):
            image = tifffile.imread(f)
            condition, div, dpi = get_metadata(f)
            
            img = image[channel]
            mask = DetectionWavelets(img, J_list=(3,4), scale_threshold=2.0).computeDetection()
            m, M = np.quantile(img, [0.0001, 0.9999])
            img = (img - m) / (M - m)
            img = np.clip(img, 0, 1)
            ys = np.arange(0, img.shape[0] - 224, 224)
            xs = np.arange(0, img.shape[1] - 224, 224)
            for y in ys:
                for x in xs:
                    crop = img[y:y+224, x:x+224] 
                    crop_mask = mask[y:y+224, x:x+224] 
                    foreground = np.count_nonzero(crop_mask)
                    pixels = 224 * 224 
                    ratio = foreground / pixels 
                    if ratio <= THRESHOLD:
                        continue
                    else:
                        added_files.append(f)
                        num_crops += 1
                        buffer = io.BytesIO() 
                        np.savez(
                            buffer,
                            image=crop,
                            mask=crop_mask,
                            condition=condition,
                            div=div,
                            dpi=dpi,
                            min_value=m,
                            max_value=M
                        )
                        buffer.seek(0)
                        name = f.split("/")[-1].split(".")[0]
                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(buffer.getbuffer())
                        handle.addfile(tarinfo=tarinfo, fileobj=buffer)
    print(f"Number of crops: {num_crops}")
    print(f"Number of large images: {len(np.unique(added_files))}")


if __name__=="__main__":
    train_files, valid_files, test_files = load_files(path=args.dataset_path, batch_id=args.batch_id, condition=args.condition)
    process_files(train_files, f"{args.dataset_path}/{args.condition}-262-{args.channel}-train")
    print("--- Processed train files ---\n")
    process_files(valid_files, f"{args.dataset_path}/{args.condition}-262-{args.channel}-valid")
    print("--- Processed valid files ---\n")
    process_files(test_files, f"{args.dataset_path}/{args.condition}-262-{args.channel}-test")
    print("--- Processed test files ---\n")
    
